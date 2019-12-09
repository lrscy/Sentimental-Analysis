import src.settings as settings
from src.parser import args
from src.data_processor import *
from transformers import BertModel, BertTokenizer
from src.UtilityFunction import GetNounWordsDictionary
from src.positiveandnegativewordsdictionary import positive_words, negative_words


def relation_extractor(args):
    # Initial parameters and build directories
    settings.USE_CUDA = args.use_cuda

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Decide output stream
    if args.shell_print == 'file':
        settings.SHELL_OUT_FILE = open(args.output_dir + 'shell_out', 'a+',
                                       encoding='utf-8')
    else:
        settings.SHELL_OUT_FILE = sys.stdout

    # Build model and data reader/processor
    ''' Shift to your own tokenizer, processor, and reader '''
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_dir + args.bert_file + '-vocab.txt')
    processor = BDProcessor(tokenizer, args.max_seq_length)
    reader = BDReader(args.batch_size)

    model = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    model = model.cuda() if settings.USE_CUDA else model
    model.eval()

    def get_whole_word(text, pos):
        st = pos
        while st >= 0 and text[st][:2] == "##":
            st -= 1
        word = text[st]
        st += 1
        while st < len(text):
            tkn = text[st]
            if tkn[:2] != "##":
                break
            word += tkn[2:]
            st += 1
        return word

    examples = reader.get_test_examples(args.data_dir)  # Change method name
    results = []
    st_example = 0
    ed_example = 1000
    while st_example < len(examples):
        print("No. of processed examples:", st_example * args.batch_size, end='\r', flush=True)
        total_attentions = []
        total_ids = []
        with torch.no_grad():
            for example in examples[st_example:ed_example]:
                inputs = processor.convert_examples_to_tensor(example)
                attentions = model(*inputs)[-1][6]  # get attentions at layer 5 need to adjust the attention layer
                attentions = torch.sum(attentions, 1)
                token_ids = inputs[0].cpu().numpy()
                total_attentions.append(attentions.cpu())
                total_ids.append(token_ids)

        for example, attentions, token_ids in zip(examples[st_example:ed_example], total_attentions, total_ids):
            noun_dict = GetNounWordsDictionary(example, args)
            for bid in range(args.batch_size):
                pad_pos = 0
                while pad_pos < token_ids.shape[1] and token_ids[bid, pad_pos] != 0:
                    pad_pos += 1
                example_attention = attentions[bid, 1:pad_pos - 1, 1:pad_pos - 1]  # remove [CLS] and [SEP]
                example_token_ids = token_ids[bid, 1:pad_pos - 1]  # remove [CLS] and [SEP]
                example_text = processor.tokenizer.convert_ids_to_tokens(example_token_ids)

                # print(noun_dict)
                # print(example_text)

                converted_attentions = []
                converted_text = []
                for i, token in enumerate(example_text):
                    if token[:2] == "##":
                        converted_text[-1] += token[2:]
                        converted_attentions[-1] += example_attention[i, :]
                    else:
                        converted_text.append(token)
                        converted_attentions.append(example_attention[i, :])

                example_result = {}
                for sent_dict in noun_dict[bid]:
                    for k, v in sent_dict.items():
                        sorted_idx = torch.sort(converted_attentions[v])[1]
                        cnt = 3
                        for idx in sorted_idx:
                            word = get_whole_word(example_text, idx)
                            if word in positive_words:
                                if k not in example_result:
                                    example_result[k] = [0, 0]  # +, -
                                example_result[k][0] += 1
                                cnt -= 1
                            if word in negative_words:
                                if k not in example_result:
                                    example_result[k] = [0, 0]  # +, -
                                example_result[k][1] += 1
                                cnt -= 1
                            if cnt == 0:
                                break
                results.append(example_result)

        st_example += 1000
        ed_example += 1000
        ed_example = min(ed_example, len(examples))

    return results


if __name__ == "__main__":
    lr = 2e-5
    ep = 5
    dp = 0.5
    b = 16
    s = 512
    wp = 0.1
    run = 1
    EMBED_DIR = "../data/bert-embedding/"
    DATA_DIR = "../data/black_decker/"
    args.data_dir = DATA_DIR
    args.bert_dir = EMBED_DIR
    args.bert_file = "bert-base-cased"
    args.do_predict = True
    args.learning_rate = lr
    args.epoch = ep
    args.use_cuda = True
    args.batch_size = b
    args.max_seq_length = s
    args.dropout = dp
    args.output_dir = "../results/re_output_lr" + str(lr) + "_ep" + str(ep) + "_dp" + str(dp) + "_b " + str(b) + "_s" \
                      + str(s) + "_wp" + str(wp) + "_run" + str(run) + "/"
    args.shell_print = True
    args.suffix = "last"
    args.multi_gpu = True
    res = relation_extractor(args)
    for d in res:
        print(d)
