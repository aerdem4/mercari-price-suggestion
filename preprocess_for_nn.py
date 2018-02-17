import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer


LIMIT = 2


def cut(x, voc):
    return " ".join([w if w in voc else "rareword" for w in x.split(" ")])


def number_preprocess(x):
    for d in re.findall(r"\d\d+", x):
        x = x.replace(d, "numericword")
    return x


class NNPreprocessor(object):
    def __init__(self):
        self.tok_raw = Tokenizer()
        self.le = {}
        self.cat_cols = ["brand_name", "subcat_0", "subcat_1", "subcat_2"]
        self.cat_vocab = {}
        for cat in self.cat_cols:
            self.le[cat] = LabelEncoder()
        self.freqs = {}
        self.max_freqs = {}
        self.voc = None

    def fit_transform(self, df):
        df["name"] = df["name"].apply(number_preprocess)
        df["item_description"] = df["item_description"].apply(number_preprocess)

        for cat in self.cat_cols:
            voc = df[cat].value_counts()
            voc = set(voc[voc >= LIMIT].index)
            self.cat_vocab[cat] = voc
        for cat in self.cat_cols:
            df[cat] = df[cat].apply(lambda x: x if x in self.cat_vocab[cat] else "rarecategory")
            df[cat] = self.le[cat].fit_transform(df[cat])

        cv = CountVectorizer(token_pattern="\w+", min_df=LIMIT)
        cv.fit(df["name"])
        name_voc = cv.vocabulary_
        cv = CountVectorizer(token_pattern="\w+", min_df=LIMIT)
        cv.fit(df["item_description"])
        desc_voc = cv.vocabulary_
        self.voc = set(name_voc).union(set(desc_voc))

        df["name"] = df["name"].apply(lambda x: cut(x, self.voc))
        df["item_description"] = df["item_description"].apply(lambda x: cut(x, self.voc))

        print("Transforming text data to sequences...")
        raw_text = np.hstack([df["name"].values, df["item_description"].values])

        print("Fitting tokenizer...")
        self.tok_raw.fit_on_texts(raw_text)

        print("Transforming text to sequences...")
        df['seq_item_description'] = self.tok_raw.texts_to_sequences(df["item_description"].values)
        df['seq_name'] = self.tok_raw.texts_to_sequences(df["name"].values)

        WC = max(self.tok_raw.word_index.values())

        for col in ["name_ori", "item_description_ori"]:
            f_col = col + "_freq"
            self.freqs[col] = df.groupby(col)["train_id"].count().reset_index()
            self.freqs[col].columns = [col, f_col]
            df = pd.merge(df, self.freqs[col], how="left", on=col)
            df[f_col] = df[f_col] - 1
            self.max_freqs[col] = df[f_col].max()
            df[f_col] = df[f_col] / self.max_freqs[col]

        return df, WC

    def transform(self, df):
        df["name"] = df["name"].apply(number_preprocess)
        df["item_description"] = df["item_description"].apply(number_preprocess)

        for cat in self.cat_cols:
            df[cat] = df[cat].apply(lambda x: x if x in self.cat_vocab[cat] else "rarecategory")
            df[cat] = self.le[cat].transform(df[cat])

        df["name"] = df["name"].apply(lambda x: cut(x, self.voc))
        df["item_description"] = df["item_description"].apply(lambda x: cut(x, self.voc))

        df['seq_item_description'] = self.tok_raw.texts_to_sequences(df["item_description"].values)
        df['seq_name'] = self.tok_raw.texts_to_sequences(df["name"].values)

        for col in ["name_ori", "item_description_ori"]:
            f_col = col + "_freq"
            df = pd.merge(df, self.freqs[col], how="left", on=col)
            df[f_col] = df[f_col].fillna(0)
            df[f_col] = df[f_col] / (self.max_freqs[col] + 1)

        return df
