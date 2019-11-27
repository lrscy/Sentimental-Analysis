import pandas as pd
import numpy as np

def RemoveBrandsandTexts(data): #this data should be original data.
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

    return data






