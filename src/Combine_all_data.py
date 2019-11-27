import pandas as pd


header = pd.read_csv("../data/black_decker/header.csv", header=None)
header = header.values[0]
# print(header)

df = pd.read_csv("../data/black_decker/000000_0", sep="~!", header=None)
for i in range(1,19):
    if i < 10:
        temp = pd.read_csv("../data/black_decker/00000"+str(i)+"_0",sep="~!",
                           header=None)
    else:
        temp = pd.read_csv("../data/black_decker/0000" + str(i) + "_0", sep="~!",
                           header=None)
    df = pd.concat([df, temp], axis=0)
df = df.reset_index(drop=True)
df.columns = header
df.to_csv("../data/black_decker/total_data.csv", index=False)
