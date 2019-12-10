import os
os.chdir("/Users/xuanshen/PycharmProjects/CS6140-Project")
import ast
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

with open("results/re_output_lr2e-05_ep5_dp0.5_b16_s512_wp0.1_run1/relationships.txt", "r") as f:
    data_results = ast.literal_eval(f.read())

data = pd.read_csv("data/test.csv")

data_brand_col = data['brand'].to_list()
data_brand_set = list(set(data_brand_col))

total_dic = []
for e in data_brand_set:
    temp_dic = {}
    temp_data = data[data['brand'] == e]
    temp_idx = temp_data.index.to_list()
    for ele in temp_idx:
        temp_dic_of_res = data_results[ele]
        if len(temp_dic_of_res) == 0:
            continue
        for key in temp_dic_of_res:
            if key in temp_dic:
                temp_dic[key][0] += temp_dic_of_res[key][0]
                temp_dic[key][1] += temp_dic_of_res[key][1]
            else:
                temp_dic[key] = temp_dic_of_res[key]
    total_dic.append(temp_dic)

str_text = ''
for e in list(total_dic[0].keys()):
    str_text = str_text + e + ','
wordcloud_show = WordCloud(max_font_size=50, max_words=100, background_color='white').generate(str_text)
plt.figure()
plt.imshow(wordcloud_show, interpolation="bilinear")
plt.axis("off")
plt.show()



