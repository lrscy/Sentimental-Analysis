
import pandas as pd
import numpy as np




header = pd.read_csv("/Users/xuanshen/PycharmProjects/CS6140-Project/data/Data/header.csv", header=None)
header = header.values[0]
# print(header)

df = pd.read_csv("/Users/xuanshen/PycharmProjects/CS6140-Project/data/Data/000000_0", sep="~!", header=None)
for i in range(1,19):
    if i < 10:
        temp = pd.read_csv("/Users/xuanshen/PycharmProjects/CS6140-Project/data/Data/00000"+str(i)+"_0",sep="~!",
                           header=None)
    else:
        temp = pd.read_csv("/Users/xuanshen/PycharmProjects/CS6140-Project/data/Data/0000" + str(i) + "_0", sep="~!",
                           header=None)
    df = pd.concat([df, temp], axis=0)
df = df.reset_index(drop=True)
df.columns = header
df.to_csv("/Users/xuanshen/PycharmProjects/CS6140-Project/data/total_data.csv", index=False)
