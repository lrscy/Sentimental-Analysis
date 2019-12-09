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
        ed = pos
        while st >= 0 and text[st][:2] == "##":
            st -= 1
        while ed + 1 < len(text) and text[ed + 1][:2] == "##":
            ed += 1
        word = ""
        for j in range(st, ed + 1):
            tkn = text[j]
            word += tkn if tkn[:2] != "##" else tkn[2:]
        return word

    examples = reader.get_test_examples(args.data_dir)  # Change method name
    results = []
    with torch.no_grad():
        for example in examples[:10]:
            noun_dict = GetNounWordsDictionary(example, args)
            inputs = processor.convert_examples_to_tensor(example)
            attentions = model(*inputs)[-1][6]  # get attentions at layer 5 need to adjust the attention layer
            attentions = torch.sum(attentions.squeeze(0), 0)  # remove [CLS] and [SEP]
            token_ids = inputs[0].squeeze(0).cpu().numpy()
            pad_pos = 0
            while pad_pos < token_ids.shape[0] and token_ids[pad_pos] != 0:
                pad_pos += 1
            attentions = attentions[1:pad_pos - 1, 1:pad_pos - 1]  # remove [CLS] and [SEP]
            token_ids = token_ids[1:pad_pos - 1]  # remove [CLS] and [SEP]
            example_text = processor.tokenizer.convert_ids_to_tokens(token_ids)

            # print(noun_dict)
            # print(example_text)

            converted_attentions = []
            converted_text = []
            for i, token in enumerate(example_text):
                if token[:2] == "##":
                    converted_text[-1] += token[2:]
                    converted_attentions[-1] += attentions[i, :]
                else:
                    converted_text.append(token)
                    converted_attentions.append(attentions[i, :])

            example_result = {}
            for sent_dict in noun_dict[0]:
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

    return results


if __name__ == "__main__":
    lr = 2e-5
    ep = 5
    dp = 0.5
    b = 1
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
