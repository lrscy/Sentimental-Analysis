import src.settings as settings
from .parser import args
from .data_processor import *
from transformers import BertModel, BertTokenizer, AdamW


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

    model = BertModel.from_pretrained(
        args.bert_dir + args.bert_file + '.bin', output_attentions=True)
    if args.multi_gpu:
        model = nn.DataParallel(model)
    model = model.cuda() if settings.USE_CUDA else model
    model.eval()

    examples = reader.get_test_examples(args.data_dir)  # Change method name
    with torch.no_grad():
        for example in examples:
            inputs = processor.convert_examples_to_tensor(example)
            attentions = model(*inputs)[-1]
            pass


if __name__ == "__main__":
    relation_extractor(args)