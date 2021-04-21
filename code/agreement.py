import pandas as pd

spacy_res = pd.read_csv("../data/all_meta.csv")
print("SpaCy found {} gendered words".format(spacy_res.shape[0]))
lal_res = pd.read_csv("../data/all_lal.csv")
print("LAL found {} gendered words".format(lal_res.shape[0]))

all = spacy_res.append(lal_res)
all = all.drop_duplicates()
print("Overall there are {} unique sentences with gendered words".format(all.shape[0]))

all.to_csv("all_merged.csv", index=False)


