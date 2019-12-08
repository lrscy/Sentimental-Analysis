import os
import ast
import csv
import math
import torch
import numpy as np
import src.settings as settings
import pandas
from src.utils import *
from src.settings import *


class InputExample(object):
    """Constructs an InputExample
  
    Args:
      text_a: 2-D list. Untokenized sentences of sequence a.
      labels: 2-D list. One-hot labels correspond to
              each sentence in context.
    """

    def __init__(self, text_a):
        self.text_a = text_a


class DataReader(object):
    """
    Base class for data converters for sequence classification
    data sets.
    """

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        data = pandas.read_csv(input_file)
        return data['text'].to_list()
#        with open(input_file, "r", encoding='utf-8') as f:
#            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
#            next(reader)
#            lines = []
#            for line in reader:
#                lines.append(line)
#            return lines


class BDReader(DataReader):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'train.csv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'dev.csv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, 'test.csv')), 'test')

    def get_labels(self):
        return {'0': 0, '1': 1}

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        total_examples = len(lines)

        text_a = []
        print("\rProcessed Examples: {}/{}".format(0,
                                                   total_examples),
              end='\r', file=settings.SHELL_OUT_FILE, flush=True)
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if i % 1000 == 0:
                print("\rProcessed Examples: {}/{}".format(i, total_examples),
                      end='\r', file=settings.SHELL_OUT_FILE, flush=True)
            text_a.append(convert_to_unicode(str(line)))

            if i % self.batch_size == 0:
                examples.append(
                    InputExample(text_a=text_a))
                text_a = []
        if len(text_a):
            examples.append(InputExample(text_a=text_a))
        print("\rProcessed Examples: {}/{}".format(total_examples, total_examples),
              file=settings.SHELL_OUT_FILE, flush=True)
        return examples


class BDProcessor(object):
    def __init__(self, tokenizer, max_seq_len=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def convert_examples_to_tensor(self, examples):
        # init
        length = len(examples.text_a)
        inputs_ids = np.zeros((length, self.max_seq_len), dtype=np.int64)
        token_type_ids = np.zeros_like(inputs_ids)

        for i, text_a in enumerate(examples.text_a):
            # inputs
            tokens = self.tokenizer.tokenize(text_a)[:self.max_seq_len - 2]
            segment_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens)
            # convert to ids
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens = self.tokenizer.build_inputs_with_special_tokens(tokens)
            # pad
            pad_len = self.max_seq_len - len(tokens)
            tokens.extend([0] * pad_len)
            segment_ids.extend([0] * pad_len)
            # to numpy
            inputs_ids[i, :] = tokens[:]
            token_type_ids[i, :] = segment_ids[:]
        # to tensor
        inputs_ids = torch.from_numpy(inputs_ids).long().detach()
        token_type_ids = torch.from_numpy(token_type_ids).long().detach()
        inputs_mask = (inputs_ids != 0).long().detach()
        inputs = [inputs_ids, inputs_mask, token_type_ids]

        if settings.USE_CUDA:
            inputs = [i.cuda() for i in inputs]

        return inputs

