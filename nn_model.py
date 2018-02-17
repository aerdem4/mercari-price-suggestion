from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras import backend
import tensorflow as tf


class NNModel(object):
    def __init__(self, train_df, word_count, batch_size, epochs):
        tf.set_random_seed(4)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=8)
        backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))

        self.batch_size = batch_size
        self.epochs = epochs

        self.max_name_seq = 10
        self.max_item_desc_seq = 75
        self.max_text = word_count + 1
        self.max_brand = np.max(train_df.brand_name.max()) + 1
        self.max_condition = np.max(train_df.item_condition_id.max()) + 1
        self.max_subcat0 = np.max(train_df.subcat_0.max()) + 1
        self.max_subcat1 = np.max(train_df.subcat_1.max()) + 1
        self.max_subcat2 = np.max(train_df.subcat_2.max()) + 1

    def get_nn_data(self, dataset):
        X = {
            'name': pad_sequences(dataset.seq_name, maxlen=self.max_name_seq),
            'item_desc': pad_sequences(dataset.seq_item_description, maxlen=self.max_item_desc_seq),
            'brand_name': np.array(dataset.brand_name),
            'item_condition': np.array(dataset.item_condition_id),
            'num_vars': np.array(dataset[["shipping", "strange_char", "desc_len", "name_len",
                                          "name_ori_freq", "item_description_ori_freq"]]),
            'subcat_0': np.array(dataset.subcat_0),
            'subcat_1': np.array(dataset.subcat_1),
            'subcat_2': np.array(dataset.subcat_2),
        }
        return X

    def new_rnn_model(self, X_train, lr_init=0.005, lr_fin=0.001):

        name = Input(shape=[X_train["name"].shape[1]], name="name")
        item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
        brand_name = Input(shape=[1], name="brand_name")
        item_condition = Input(shape=[1], name="item_condition")
        num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
        subcat_0 = Input(shape=[1], name="subcat_0")
        subcat_1 = Input(shape=[1], name="subcat_1")
        subcat_2 = Input(shape=[1], name="subcat_2")

        emb_text = Embedding(self.max_text, 35)
        emb_name = emb_text(name)
        emb_item_desc = emb_text(item_desc)
        emb_brand_name = Embedding(self.max_brand, 10)(brand_name)
        emb_item_condition = Embedding(self.max_condition, 10)(item_condition)
        emb_subcat_0 = Embedding(self.max_subcat0, 10)(subcat_0)
        emb_subcat_1 = Embedding(self.max_subcat1, 10)(subcat_1)
        emb_subcat_2 = Embedding(self.max_subcat2, 10)(subcat_2)

        rnn_layer1 = GaussianNoise(0.01)(GRU(10)(emb_item_desc))
        rnn_layer2 = GaussianNoise(0.01)(GRU(8)(emb_name))
        cat_layer = average([Flatten()(emb_subcat_0), Flatten()(emb_subcat_1), Flatten()(emb_subcat_2)])

        # main layers
        main_l = concatenate([Flatten()(emb_brand_name), Flatten()(emb_item_condition),
                              cat_layer, multiply([cat_layer, item_condition]),
                              rnn_layer1, rnn_layer2, Dense(8, activation="tanh")(num_vars)])

        main_l = Dropout(0.1)(Dense(512, kernel_initializer='normal', activation='relu')(main_l))
        main_l = Dropout(0.1)(Dense(256, kernel_initializer='normal', activation='relu')(main_l))
        main_l = Dropout(0.1)(Dense(128, kernel_initializer='normal', activation='relu')(main_l))
        main_l = Dropout(0.1)(Dense(64, kernel_initializer='normal', activation='relu')(main_l))

        output = Dense(1, activation="linear")(main_l)

        model = Model([name, item_desc, brand_name, item_condition,
                       num_vars, subcat_0, subcat_1, subcat_2], output)

        steps = int(len(X_train['name']) / self.batch_size) * self.epochs
        lr_decay = (lr_init / lr_fin) ** (1 / (steps - 1)) - 1
        optimizer = Adam(lr=lr_init, decay=lr_decay)

        model.compile(loss='mse', optimizer=optimizer)

        return model
