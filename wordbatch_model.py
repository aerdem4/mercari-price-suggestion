import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL


class WordBatchModel(object):
    def __init__(self):
        self.wb_desc = None
        self.desc_indices = None
        self.cv_name, self.cv_name2 = None, None
        self.cv_cat0, self.cv_cat1, self.cv_cat2 = None, None, None
        self.cv_brand = None
        self.cv_condition = None
        self.cv_cat_brand = None
        self.desc3 = None
        self.model = None

    def train(self, df):

        self.wb_desc = wordbatch.WordBatch(None,
                                           extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                "idf": None}), procs=8)
        self.wb_desc.dictionary_freeze = True
        X_desc = self.wb_desc.fit_transform(df['item_description'])
        self.desc_indices = np.array(np.clip(X_desc.getnnz(axis=0) - 1, 0, 1), dtype=bool)
        X_desc = X_desc[:, self.desc_indices]

        self.cv_name = CountVectorizer(min_df=2, ngram_range=(1, 1),
                                       binary=True, token_pattern="\w+")
        X_name = 2 * self.cv_name.fit_transform(df['name'])
        self.cv_name2 = CountVectorizer(min_df=2, ngram_range=(2, 2),
                                        binary=True, token_pattern="\w+")
        X_name2 = 0.5 * self.cv_name2.fit_transform(df['name'])

        self.cv_cat0 = CountVectorizer(min_df=2)
        X_category0 = self.cv_cat0.fit_transform(df['subcat_0'])
        self.cv_cat1 = CountVectorizer(min_df=2)
        X_category1 = self.cv_cat1.fit_transform(df['subcat_1'])
        self.cv_cat2 = CountVectorizer(min_df=2)
        X_category2 = self.cv_cat2.fit_transform(df['subcat_2'])

        self.cv_brand = CountVectorizer(min_df=2, token_pattern=".+")
        X_brand = self.cv_brand.fit_transform(df['brand_name'])

        self.cv_condition = CountVectorizer(token_pattern=".+")
        X_condition = self.cv_condition.fit_transform((df['item_condition_id'] + 10 * df["shipping"]).apply(str))

        df["cat_brand"] = [a + " " + b for a, b in zip(df["category_name"], df["brand_name"])]
        self.cv_cat_brand = CountVectorizer(min_df=10, token_pattern=".+")
        X_cat_brand = self.cv_cat_brand.fit_transform(df["cat_brand"])

        self.desc3 = CountVectorizer(ngram_range=(3, 3), max_features=1000, binary=True, token_pattern="\w+")
        X_desc3 = self.desc3.fit_transform(df["item_description"])

        X = hstack((X_condition,
                    X_desc, X_brand,
                    X_category0, X_category1, X_category2,
                    X_name, X_name2,
                    X_cat_brand, X_desc3)).tocsr()

        y = df["target"].values

        self.model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=X.shape[1], alpha_fm=0.02, L2_fm=0.0,
                             init_fm=0.01, D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)
        self.model.fit(X, y)

    def predict(self, df):
        X_desc = self.wb_desc.transform(df["item_description"])
        X_desc = X_desc[:, self.desc_indices]

        X_name = 2 * self.cv_name.transform(df["name"])
        X_name2 = 0.5 * self.cv_name2.transform(df["name"])

        X_category0 = self.cv_cat0.transform(df['subcat_0'])
        X_category1 = self.cv_cat1.transform(df['subcat_1'])
        X_category2 = self.cv_cat2.transform(df['subcat_2'])
        X_brand = self.cv_brand.transform(df['brand_name'])
        X_condition = self.cv_condition.transform((df['item_condition_id'] + 10 * df["shipping"]).apply(str))

        df["cat_brand"] = [a + " " + b for a, b in zip(df["category_name"], df["brand_name"])]
        X_cat_brand = self.cv_cat_brand.transform(df["cat_brand"])
        X_desc3 = self.desc3.transform(df["item_description"])

        X = hstack((X_condition,
                    X_desc, X_brand,
                    X_category0, X_category1, X_category2,
                    X_name, X_name2,
                    X_cat_brand, X_desc3)).tocsr()

        return self.model.predict(X)
