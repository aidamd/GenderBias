import glob
import json
import re
import spacy
nlp = spacy.load("en")
import pandas as pd
from collections import defaultdict
import pickle
import os
from tqdm import tqdm
import itertools


class Dataset():
    def __init__(self, input_pattern, shows, path):
        self.dictionary = self.load_liwc_dictionaries(os.path.join(path, "data", "liwc.pk"))
        self.load_agency_power(os.path.join(path, "FramesAgencyPower", "agency_power.csv"))

        self._subjects = ["nsubj", "agent"]
        self._objects = ["dobj", "iobj", "nsubjpass"]

        self.male_cat = pickle.load(open(os.path.join(path, "data", "male.pk"), "rb"))
        self.male_cat.extend(["him", "his", "himself"])
        self.female_cat = pickle.load(open(os.path.join(path, "data", "female.pk"), "rb"))
        self.female_cat.extend(["her", "hers", "herself"])

        self.shows= shows
        self.input_pattern = input_pattern
        self.data_file = os.path.join(path, "all.csv")
        self.meta_file = os.path.join(path, "all_meta.csv")
        self.final_file = os.path.join(path, "all_rearranged.csv")

    def load_liwc_dictionaries(self, liwc_path):
        print("Loading the liwc dictionary file as a pickle")
        liwc = pickle.load(open(liwc_path, "rb"))
        dictionary_re = dict()
        for cat, words in liwc.items():
            dictionary_re[cat] = list()
            for word in words:
                word = word.replace(")", "\\)").replace("(", "\\(")\
                    .replace(":", "\\:").replace(";", "\\;").replace("/", "\\/")

                if word[-1] == "*":
                    dictionary_re[cat].append(re.compile("(" + word + "\w*)"))
                else:
                    dictionary_re[cat].append(re.compile("(" + word + ")"))
        return dictionary_re

    def load_agency_power(self, file):
        print("Loading the agency and power connotations")
        connotations = pd.read_csv(file)
        self.agency_power = defaultdict(dict)
        for i, row in connotations.iterrows():
            try:
                verb = nlp(row["verb"])[0].lemma_
                self.agency_power[verb]["agency"] = row["agency"].split("_")[1]
                self.agency_power[verb]["power"] = row["power"].split("_")[1]
            except Exception:
                continue

    
    def read(self):
        print("Reading scripts")
        show_list = pd.read_excel(self.shows)["shows"].tolist()
        for count, name in enumerate(glob.glob(self.input_pattern)):
            print(name)
            df = defaultdict(list)
            file = open(name, 'rb').readlines()
            uni_re = re.compile(r"\\[a-zA-Z0-9]*")
            alpha_re = re.compile(r"^[a-zA-Z.,;!?()\'\"]")

            #pbar = tqdm(total=len(file))
            for line in tqdm(file):
                l = line.decode("utf-8", "ignore").replace('\r', '').replace('\t', '').replace('\n', '')
                l1 = uni_re.sub("", l)
                l2 = alpha_re.sub("", l1)
                try:
                    episode = json.loads(l2.replace("\\", ""))
                except Exception:
                    continue
                if episode["show"] in show_list:
                    sentences = [sent for sent in nlp(episode["script"]).sents]
                    df["Show"].extend([episode["show"] for i in range(len(sentences))])
                    df["Episode"].extend([episode["episode"] for i in range(len(sentences))])
                    df["Sentence"].extend(sentences)

            result = pd.DataFrame.from_dict(df)
            if count == 0:
                result.to_csv(self.data_file, index=False)
            else:
                result.to_csv(self.data_file, index=False, mode="a")

    def get_roles(self):
        df = pd.read_csv(self.data_file)
        new_df = list()
        complexity = list()
        ans = defaultdict(list)

        for i, row in df.iterrows():
            doc = nlp(row["Sentence"])
            doc_gender = self.get_gender_role(doc)
            if sum([len(val) for val in doc_gender.values()]) > 0:
                # vocab = self.liwc_ratio(row["Sentence"])
                # for concept in vocab:
                #    ans[concept].append(vocab[concept])
                # complexity.append(self.get_height(doc))
                for key, gender_roles in doc_gender.items():
                    for tok, agency_power in gender_roles.items():
                        new_row = {k: v for k, v in row.items()}
                        for role in doc_gender.keys():
                            new_row[role] = 0
                        new_row[key] = 1
                        for score in ["agency", "power"]:
                            new_row["token_" + score] = -agency_power[score] if "Patient" in key \
                                else agency_power[score]
                        #new_row.update(vocab)
                        new_df.append(new_row)

        #print("Removing", len(non))
        #print("Comp", len(complexity))
        #df = df.drop(non)
        print("Gendered Roles:", len(new_df))
        pd.DataFrame(new_df).to_csv(self.meta_file, index=False)

    def get_gender_role(self, sentence):
        # load male and female words, stored as pickle files
        roles = defaultdict(list)
        if isinstance(sentence, str):
            sentence = nlp(sentence)

        verbs = {tok : {"agency": self.get_agency(tok), "power": self.get_power(tok) }
                 for tok in sentence if tok.pos_ == "VERB"}

        toks = dict()
        toks["Agent"] = [tok for tok in sentence if (tok.dep_ in self._subjects)]
        toks["Patient"] = [tok for tok in sentence if (tok.dep_ in self._objects)]
        toks["Other"] = [tok for tok in sentence if (tok.dep_ not in self._subjects
                                                and tok.dep_ not in self._objects)]
        for role in ["Agent", "Patient", "Other"]:
            female, male = list(), list()
            for sub in toks[role]:
                zir_sub = str(sub).split()
                female.extend([zir for zir in zir_sub if zir in self.female_cat])
                male.extend([zir for zir in zir_sub if zir in self.male_cat])

            roles["Female_" + role] = {gen: {"agency": 0, "power": 0} for gen in female}
            roles["Male_" + role] = {gen: {"agency": 0, "power": 0} for gen in male}

            if role == "Other":
                continue

            roles["Female_" + role].update({gen: verbs[verb] for gen, verb in
                                        itertools.product(female, list(verbs.keys())) if gen in [str(x) for x in verb.children]})
            roles["Male_" + role].update({gen: verbs[verb] for gen, verb in
                                      itertools.product(male, list(verbs.keys())) if gen in [str(x) for x in verb.children]})
            """
            for gen in female:
                for verb in verbs.keys():
                    if gen in [str(x) for x in verb.children]:
                        print(sentence, role, verb, verbs[verb], gen, roles["Female_" + role][gen])
            for gen in male:
                for verb in verbs.keys():
                    if gen in [str(x) for x in verb.children]:
                        print(sentence, role, verb, verbs[verb], gen, roles["Male_" + role][gen])
            """

        return roles

    def get_power(self, verb):
        power_map = {"agent": 1,
                     "equal": 0,
                     "theme": -1}
        try:
            power = power_map[self.agency_power[verb.lemma_]["power"]]
        except Exception:
            power = 0
        return power

    def get_agency(self, verb):
        agency_map = {"pos": 1,
                "equal": 0,
                "neg": -1}
        try:
            agency = agency_map[self.agency_power[verb.lemma_]["agency"]]
        except Exception:
            agency = 0
        return agency

    def liwc_ratio(self, sentence):
        vector = dict()
        for cat in self.dictionary.keys():
            count = 0
            for reg in self.dictionary[cat]:
                x = len(re.findall(reg, sentence))
                if x > 0:
                    count += x
            if len(sentence.split()) == 0:
                vector[cat] = 0
            else:
                vector[cat] = 100 * float(count) / float(len(sentence.split()))
        return vector


    def get_height(self, doc):
        if len(doc.sents) > 1:
            print("More than one sentences")
        return self.get_depth(doc.sents[0].root)

    def get_depth(self, token, depth=0):
        if len([i for i in token.children]) > 0:
            return max([self.get_depth(child, depth) + 1 for child in token.children])
        else:
            return 0

    def rearange(self):
        df = pd.read_csv(self.meta_file)
        gender_role = ["Female_Agent", "Female_Patient",
                       "Male_Agent", "Male_Patient",
                       "Female_Other", "Male_Other"]
        new_lines = list()
        drop = list()

        for i, row in df.iterrows():
            if sum([row[x] for x in gender_role]) > 1:
                drop.append(i)
                for col in gender_role:
                    if row[col] > 0:
                        row_new = {key: row[key] for key in row.keys()}
                        for c in gender_role:
                            if c != col:
                                row_new[c] = 0
                            else:
                                row_new[c] = row[c]
                        new_lines.append(row_new)
        print("Before rearrange:", df.shape)
        df = df.drop(drop)
        print("Sentences to keep:",  df.shape)
        new = pd.DataFrame(new_lines)
        print("Adding", new.shape, "duplicates")
        df = df.append(new)
        print("Final shape:", df.shape)
        df.to_csv(self.final_file, index=False)
