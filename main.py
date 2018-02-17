import string
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wordbatch_model import WordBatchModel
from preprocess_for_nn import NNPreprocessor
from nn_model import NNModel


def secure(df):
    df["category_name"] = df["category_name"].fillna(value="missing").apply(str)
    df["name"] = df["name"].fillna(value="missing").apply(str)
    df["brand_name"] = df["brand_name"].fillna(value="missing").apply(str)
    df["item_description"] = df["item_description"].fillna(value="missing").apply(str)
    df["item_condition_id"] = df["item_condition_id"].fillna(value=1).apply(int)
    df["shipping"] = df["shipping"].fillna(value=0).apply(int)
    return df


def word_count(text):
    if text == 'No description yet':
        return 0
    else:
        return len(text.lower().split(" "))


def extract_len_feature(df):
    df['desc_len'] = df['item_description'].apply(word_count)
    df['name_len'] = df['name'].apply(word_count)
    return df


def split_cat(text):
    cats = text.split("/")
    if len(cats) > 3:
        cats = [cats[0], cats[1], " ".join(cats[2:])]
    while len(cats) < 3:
        cats.append("missing")
    return tuple(cats)


def plural(x):
    if len(x) > 4 and x[-1] == "s":
        return x[:-1] + " plural"
    else:
        return x


def normalize_text(translator, x):
    x = x.replace("+", " plus ").replace("&", " and ").replace("$", " dollars ")
    return " ".join([plural(x) for x in re.findall(r"\w+", x.translate(translator).lower())])


def preprocess(df):
    df["name_ori"] = df["name"].values
    df["item_description_ori"] = df["item_description"].values

    df["strange_char"] = df["name"].apply(lambda x: max([ord(c) for c in list(x)]) > 1000)
    df["item_description"] = df["item_description"].replace('No description yet', "missing2")

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    df["name"] = df["name"].apply(lambda x: normalize_text(translator, x))
    df["item_description"] = df["item_description"].apply(lambda x: normalize_text(translator, x))
    return df


def load_test():
    for df in pd.read_csv('../input/test.tsv', sep='\t', chunksize=350000):
        yield df

if __name__ == "__main__":
    np.random.seed(0)
    batch_size = 1536
    epochs = 3

    train_df = pd.read_table('../input/train.tsv', sep='\t')
    train_df = secure(train_df)
    train_df = extract_len_feature(train_df)
    train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = zip(*train_df['category_name'].apply(split_cat))
    train_df["target"] = np.log1p(train_df.price)
    train_df = preprocess(train_df)
    train_df, val_df = train_test_split(train_df, random_state=123, train_size=0.99)

    wbm = WordBatchModel()
    wbm.train(train_df)
    predsFM_val = wbm.predict(val_df)

    nnp = NNPreprocessor()
    train_df, WC = nnp.fit_transform(train_df)
    val_df = nnp.transform(val_df)

    nnm = NNModel(train_df=train_df, word_count=WC, batch_size=batch_size, epochs=epochs)
    X_train = nnm.get_nn_data(train_df)
    Y_train = train_df.target.values.reshape(-1, 1)

    X_val = nnm.get_nn_data(val_df)
    Y_val = val_df.target.values.ravel()

    rnn_model = nnm.new_rnn_model(X_train)
    rnn_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=1)
    Y_val_preds_rnn = rnn_model.predict(X_val, batch_size=batch_size).ravel()

    rmsle = lambda y, y_pred: np.sqrt(np.mean(np.square(y_pred - y)))
    print("Evaluating the model on validation data...")
    print(" RMSLE error:", rmsle(Y_val, predsFM_val))
    print(" RMSLE error:", rmsle(Y_val, Y_val_preds_rnn))
    print(" RMSLE error:", rmsle(Y_val, 0.4 * Y_val_preds_rnn + 0.6 * predsFM_val))

    # batch prediction in order to avoid memory errors
    test_ids = np.array([], dtype=np.int32)
    preds = np.array([], dtype=np.float32)
    for test_df in load_test():
        test_df = secure(test_df)
        test_df = extract_len_feature(test_df)
        test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = zip(*test_df['category_name'].apply(split_cat))
        test_df = preprocess(test_df)

        predsFM = wbm.predict(test_df)
        test_df = nnp.transform(test_df)
        X_test = nnm.get_nn_data(test_df)
        rnn_preds = rnn_model.predict(X_test, batch_size=batch_size, verbose=1).ravel()

        preds = np.append(preds, 0.4 * rnn_preds + 0.6 * predsFM)
        test_ids = np.append(test_ids, test_df["test_id"])

    preds[preds < 0] = 0
    submission = pd.DataFrame({
        "test_id": test_ids,
        "price": np.expm1(preds),
    })
    submission.to_csv("submission.csv", index=False)
