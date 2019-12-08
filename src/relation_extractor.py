import src.settings as settings
from src.parser import args
from src.data_processor import *
from transformers import BertModel, BertTokenizer
from src.UtilityFunction import getnounwordsdictionary


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
        while text[st][:2] == "##":
            st -= 1
        while ed + 1 < len(text) and text[ed + 1][:2] == "##":
            ed += 1
        word = ""
        for j in range(st, ed + 1):
            tkn = example_text[j]
            word += tkn if tkn[:2] != "##" else tkn[2:]
        return word

    examples = reader.get_test_examples(args.data_dir)  # Change method name
    results = []
    with torch.no_grad():
        for example in examples[:2]:
            noun_dict = getnounwordsdictionary(example.text_a, args)
            inputs = processor.convert_examples_to_tensor(example)
            attentions = model(*inputs)[-1][5]  # get attentions at layer 5 need to adjust the attention layer
            attentions = torch.sum(attentions.squeeze(0), 0)[1:-1, 1:-1]  # remove [CLS] and [SEP]
            token_ids = inputs[0].squeeze(0).cpu().numpy()
            pad_pos = 0
            while pad_pos < token_ids.shape[0] and token_ids[pad_pos] != 0:
                pad_pos += 1
            token_ids = token_ids[:pad_pos]
            example_text = processor.tokenizer.convert_ids_to_tokens(token_ids)

            print(noun_dict)
            print(example_text)

            example_result = {}
            pre_token = ""
            pre_pos = 0
            for i, token in enumerate(example_text):
                if token[:2] == "##":
                    token = pre_token + token[2:]
                    pre_token = token
                    if i < len(example_text) and example_text[i + 1][:2] != "##":
                        continue
                    if token in noun_dict:
                        # policy: add all tokens' attention
                        pos = torch.max(torch.sum(attentions[pre_pos: i], 0)).item()
                        example_result[token] = get_whole_word(example_text, pos)
                else:
                    pre_token = token
                    pre_pos = i
                    if token in noun_dict:
                        pos = torch.argmax(attentions[i]).item()
                        example_result[token] = get_whole_word(example_text, pos)
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
    print(res)
