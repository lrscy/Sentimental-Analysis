import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from src.positiveandnegativewordsdictionary import positive_words, negative_words
from transformers import BertTokenizer

def RemoveBrandsandTexts(data): # this data should be original data.
    # we need to lower all the text to analytics
    print("Lower all of the text for convenience of analytics.")
    text_columns = data['text'].to_list()
    for i in range(len(text_columns)):
        text_columns[i] = str(text_columns[i]).lower()
    text_columns_df = pd.DataFrame(text_columns, columns=['text'])
    data['text'] = text_columns_df['text']
    print("Done.")

    # we need to remove some brands that contain a little information.
    print("Remove the brands that contain a little information.")
    brand = data['brand'].value_counts()
    brand_keys = np.array(brand.keys().to_list())
    brand_counts = np.array(brand.to_list())
    brand_keys_countbiggerthan10000 = brand_keys[brand_counts >= 10000]
    record_idx = []
    for i in range(len(data)):
        if i % (int(len(data) / 10)) == 0:
            print("-", end='')
        if i == len(data) - 1:
            print('-')
        if data.loc[i, 'brand'] not in brand_keys_countbiggerthan10000:
            record_idx.append(i)
    data.drop(record_idx, inplace=True)
    data.reset_index(inplace=True)
    print("Done.")

    # we need to remove those data with text "Rating provided by a verified purchaser",
    # because we can not get any information from this text.
    # we first split it from the original dataset.
    print("Remove those data with text Rating provided by a verified purchaser")
    data_notext = data[data['text'] == 'Rating provided by a verified purchaser']
    data_notext.reset_index(inplace=True)
    data = data[data['text'] != 'Rating provided by a verified purchaser']
    data.reset_index(inplace=True)
    print("Done.")

    data.drop(['level_0', 'index'], axis=1)

    return data


def PositveAndNegativeWordsAnalytics(data): # this data should be the data after remove brands and texts.
    # we need to get the dictionary of every text.
    print("Get the dictionary of every text.")
    text_columns = data['text'].to_list()
    text_every_dictionary = []  # get an dictionary from every text and put into the list
    for i in range(len(text_columns)):
        if i % (int(len(data) / 10)) == 0:
            print("-", end='')
        if i == len(data) - 1:
            print('-')
        temp_sens = nltk.sent_tokenize(text_columns[i])
        temp_word = [nltk.word_tokenize(sentence) for sentence in temp_sens]
        temp_dict = {}
        for ele in temp_word:
            for e in ele:
                # the word should not be some symbols like ',' '.'
                temp_symbol = string.punctuation + '\''
                if e not in temp_symbol:
                    # we need to delete the stop words in our dictionary to make it cleaner.
                    temp_stopwords_english = set(stopwords.words('english'))
                    if e not in temp_stopwords_english:
                        if e not in temp_dict:
                            temp_dict[e] = 1
                        else:
                            temp_dict[e] += 2
        text_every_dictionary.append(temp_dict)
    print("Done.")

    # we need to get the dictionary of all texts whose stopwords were deleted
    print("Get the dictionary of all text.")
    total_dictionary = {}
    for e in text_every_dictionary:
        for ele in e:
            if ele not in total_dictionary:
                total_dictionary[ele] = e[ele]
            else:
                total_dictionary[ele] += e[ele]
    print("Done.")

    # we need to get the positive and negative words in the total dictionary according to the
    # positive and negative words dictionary got from the github:
    # https://github.com/shekhargulati/sentiment-analysis-python/tree/master/opinion-lexicon-English
    print("Get the positive and negative words from the total dictionary seperately.")
    total_dictionary_positive = {}
    total_dictionary_negative = {}
    for e in total_dictionary:
        if e in positive_words:
            total_dictionary_positive[e] = total_dictionary[e]
        elif e in negative_words:
            total_dictionary_negative[e] = total_dictionary[e]
    print("Done.")

    return total_dictionary_positive, total_dictionary_negative


def GetNounWordsDictionary(data, args):
    # get the noun words dictionary list, the list contains the noun word and the index of it in its sentence.
#    print('Get the dictionary of noun words.')
    noun_dictionary_list = []
    # use bert tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(
        args.bert_dir + args.bert_file + '-vocab.txt')

    def tokenize(text):
        tokens = bert_tokenizer.tokenize(text)[1: -1]
        converted_tokens = []
        for token in tokens:
            if token[:2] == "##":
                converted_tokens[-1] += token[2:]
            else:
                converted_tokens.append(token)
        return converted_tokens

#    data_text = data['text'].to_list()
    for i in range(len(data)):
        # if i % (int(len(data) / 10)) == 0:
        #     print("-", end='')
        # if i == len(data) - 1:
        #     print('-')
        record_length_start = 0
        temp_dic_total = []
        temp = data[i]
#        temp = data_text[i]
        temp_sens = nltk.sent_tokenize(temp)
        temp_word = [tokenize(sentence) for sentence in temp_sens]

        for k in range(len(temp_word)):
            temp_dic = {}
            temp_postag = nltk.pos_tag(temp_word[k])
            for j in range(len(temp_postag)):
                if temp_postag[j][1][:2] == 'NN':
                    temp_dic[temp_postag[j][0]] = j + record_length_start
            temp_dic_total.append(temp_dic)
            record_length_start += len(temp_word[k])

        noun_dictionary_list.append(temp_dic_total)
#    print('Done.')
    return noun_dictionary_list


if __name__ == "__main__":
    data = pd.read_csv("../data/black_decker/total_data.csv")
    data = RemoveBrandsandTexts(data)
    data.to_csv("../data/black_decker/test.csv", index=False)
