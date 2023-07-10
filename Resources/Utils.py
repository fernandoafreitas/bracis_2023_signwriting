from sklearn import preprocessing
from transformers import AutoTokenizer
import random
import numpy as np
import tensorflow.compat.v1 as tf
import pandas as pd
import os

class Utils:

    def __init__(self):
        self.c = {'EMB_DIM': 512,
     'PADDING': 'same',
     'dataset': 'details',
     'FFN_UNITS': 512,
     'NB_EPOCHS': 200,
     'size_desc': 4,
     'NB_FILTERS': 150,
     'batch_size': 32,
     'repet_test': 3,
     'repet_train': 3,
     'DROPOUT_RATE': 0.1}

    def clean_tweet(self,tweet):
        # tweet = BeautifulSoup(tweet, 'lxml').get_text()
        # tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
        # tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
        # tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
        # tweet = re.sub(r" +", ' ', tweet)
        return tweet

    def encode_sentence(self, sent, tokenizer):
        return tokenizer.encode(sent).ids

    def iteracoesDataset(self, train_dataset, test_dataset, data, tokenizer, c):
        # reseta os indices das linhas
        train_dataset.reset_index(inplace=True, drop=True)
        test_dataset.reset_index(inplace=True, drop=True)

        # transforma os labels em inteiros
        le = preprocessing.LabelEncoder()
        le.fit(data.label.values)

        train_data_labels = train_dataset.label.values
        test_data_labels = test_dataset.label.values

        # train_dataset.loc[:,'label_idx'] = train_data_labels
        train_dataset = train_dataset.assign(label_idx=train_data_labels)
        # test_dataset.loc[:,'label_idx'] = test_data_labels
        test_dataset = test_dataset.assign(label_idx=test_data_labels)

        data_labels = train_dataset.label_idx.values

        # limpa os dados caso queira retirar algum caracter especial
        train_dataset_clean = [self.clean_tweet(tweet)
                            for tweet in train_dataset.sentence]
        test_dataset_clean = [self.clean_tweet(tweet)
                            for tweet in test_dataset.sentence]



        train_inputs = [self.encode_sentence(sentence, tokenizer)
                        for sentence in train_dataset_clean]
        test_inputs = [self.encode_sentence(sentence, tokenizer)
                    for sentence in test_dataset_clean]

        train = [[sent, train_data_labels[i], len(
            sent)] for i, sent in enumerate(train_inputs)]
        test = [[sent, test_data_labels[i], len(sent)]
                for i, sent in enumerate(test_inputs)]

        random.shuffle(train)
        train.sort(key=lambda x: x[2])
        train_sorted_all = [(sent_lab[0], sent_lab[1])
                            for sent_lab in train if sent_lab[2] > c['size_desc']]

        random.shuffle(test)
        test.sort(key=lambda x: x[2])
        test_sorted_all = [(sent_lab[0], sent_lab[1])
                        for sent_lab in test if sent_lab[2] > c['size_desc']]

        train_tf = tf.data.Dataset.from_generator(lambda: train_sorted_all,
                                                output_types=(tf.int32, tf.int32))

        test_tf = tf.data.Dataset.from_generator(lambda: test_sorted_all,
                                                output_types=(tf.int32, tf.int32))
        next(iter(train_tf))
        next(iter(test_tf))

        train_tf = train_tf.padded_batch(
            c['batch_size'], padded_shapes=((None, ), ()))
        test_tf = test_tf.padded_batch(c['batch_size'], padded_shapes=((None, ), ()))

        next(iter(train_tf))
        next(iter(test_tf))

        TRAIN_NB_BATCHES = len(train_sorted_all) // c['batch_size']
        TEST_NB_BATCHES = len(test_sorted_all) // c['batch_size']

        train_tf.shuffle(TRAIN_NB_BATCHES)
        test_tf.shuffle(TEST_NB_BATCHES)
        test_dataset = test_tf.take(TEST_NB_BATCHES)
        train_dataset = train_tf.take(TRAIN_NB_BATCHES)

        next(iter(test_dataset))
        next(iter(train_dataset))
        return train_dataset, test_dataset

    def get_prediction(self,r):
        res = np.argmax(r)
        return res

    def load_db(self,language, type):
        data = pd.read_csv('Outputs/Dicts/{}_raw_{}.txt'.format(language, type),
                skip_blank_lines=True, delimiter='&', names=['label', 'sentence'])

        data.drop_duplicates(subset=['sentence'], inplace=True, keep='first')
        return data

    def selectSubGroupDatabase(self,data, n):
        data = data[data['label'].map(data['label'].value_counts()).gt(n)]
        data.reset_index(inplace=True, drop=True)
        le = preprocessing.LabelEncoder()
        le.fit(data.label.values)
        data_labels = le.transform(data.label.values)
        # data.loc[:,'label'] = data_labels.copy()
        data = data.assign(label=data_labels)
        return data

    def reset_wandb_env(self):
        exclude = {
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_API_KEY",
        }
        for k, v in os.environ.items():
            if k.startswith("WANDB_") and k not in exclude:
                del os.environ[k]
