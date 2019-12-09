from src.relation_extractor import relation_extractor
from src.parser import args
import pandas as pd
import pprint
import os


def summary(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fr = open(args.output_dir + "relationships.txt", "w+")
    data = pd.read_csv(args.data_dir + "test.csv")
    stars = data["stars"].to_list()
    results = relation_extractor(args)
    print(len(results))
    pprint.pprint(results, fr)

    tp = fp = fn = tn = 0
    for result, star in zip(results, stars):
        pos_words = neg_words = 0
        # positive and negative words count in each review
        for k, v in result.items():
            if v[0] > v[1]:
                pos_words += 1
            elif v[0] < v[1]:
                neg_words += 1
        ratio = (pos_words + 1) / (neg_words + 1)
        th_review = 2  # threshold
        th_star = 3  # threshold

        if ratio >= th_review and ratio >= th_star:
            tp += 1
        if ratio >= th_review and ratio <= th_star:
            fp += 1
        if ratio <= th_review and ratio >= th_star:
            fn += 1
        if ratio <= th_review and ratio <= th_star:
            tn += 1

    acc = (tp + tn) / (tp + fp + tn + fn)
    mis = 1 - acc
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(acc, mis, tpr, tnr)
    print(tp, fp, fn, tn)


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
    args.output_dir = "../results/re_output_lr" + str(lr) + "_ep" + str(ep) + "_dp" + str(dp) + "_b" + str(b) + "_s" \
                      + str(s) + "_wp" + str(wp) + "_run" + str(run) + "/"
    args.shell_print = True
    args.suffix = "last"
    args.multi_gpu = True
    summary(args)

