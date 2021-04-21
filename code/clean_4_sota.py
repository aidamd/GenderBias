import pandas as pd

df = pd.read_csv("/home/aida/Projects/GenderBias/data/all.csv")
print(df.shape)
with open("all_sentences.txt", "w") as f:
    for i, row in df.iterrows():
        f.write(row["Sentence"])
        f.write("\n")
