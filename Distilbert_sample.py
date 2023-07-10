from abc import ABC, abstractmethod
import os
import wandb
from transformers import DistilBertForSequenceClassification, Trainer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel, DistilBertTokenizerFast
from sklearn import preprocessing
from itertools import product
import pandas as pd
import uuid
from typing import List
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from Resources.DCNN import DCNN
import numpy as np
from Resources.Utils import Utils
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tokenizers import Tokenizer

from transformers import EarlyStoppingCallback
import torch
from transformers import TrainingArguments


class AbstractClass(ABC):

    def template_method(self, s):
        """
        O método template define o esqueleto de um algoritmo.
        """
        dataset, language, k, tokenizer, model,batch_size = s

        epochs = 200

        inputs = self.loadDatabase(language, dataset, k)
        sanitized_inputs, labels = self.preprossing(inputs, tokenizer)
        tokenizer, VOCAB_SIZE = self.loadTokenizer(tokenizer)
        train_inputs, test_inputs = self.splitData(k,sanitized_inputs,tokenizer)

        num_labels = len(inputs['label'].unique())
        hash_code = str(str(uuid.uuid1()).split('-')[0])

        for i in range(k):
            self.conectaWandb(s, i, hash_code)

            train_dataset = train_inputs[i]
            test_dataset = test_inputs[i]
            c = {
                    'batch_size': batch_size,
                    'size_desc': 4
                }
            train_dataset, test_dataset = self.iteracoesDataset(train_dataset, test_dataset, inputs, tokenizer,c)
            NB_CLASSES = num_labels

            model = self.loadModel(VOCAB_SIZE, NB_CLASSES)

            model = self.trainModel(model, train_dataset, test_dataset, batch_size, epochs,hash_code)

            self.evalModel(model, test_dataset)

            self.disconnectWandb()

    # These operations already have implementations.

    def conectaWandb(self, s, i,hash_code):
        load_dotenv()
        os.environ["WANDB_API_KEY"] = os.environ['WANDB_API_KEY']
        wandb_entity = os.environ["WANDB_ENTITY"]
        wandb_project = os.environ["WANDB_PROJECT"]

        dataset, language, k, tokenizer, model,batch_size = s

        wandb.init(project="bracis", entity=wandb_entity,
                   config={
                       'dataset': dataset,
                       'language': language,
                       'k': k,
                       'tokenizer': tokenizer,
                       'model': model
                   },
                   group=language,
                   job_type="{}-{}".format(k, hash_code),
                   name='{}-{}'.format(hash_code, i+1),
                   force=True,

                   )

    def disconnectWandb(self) -> None:
        wandb.finish()

    def loadDatabase(self, language, db_type, k) -> None:
        # Carrega o dataset
        data = pd.read_csv('Outputs/Dicts/{}_raw_{}.txt'.format(language, db_type),
                           skip_blank_lines=True, delimiter='&', names=['label', 'sentence'])
        data.drop_duplicates(subset=['sentence'], inplace=True, keep='first')

        # Pega todas as entradas que possuem ao menos repet_train repetições
        data = data[data['label'].map(
            data['label'].value_counts()).gt(k-1)]
        data.reset_index(inplace=True, drop=True)

        return data

    def loadTokenizer(self, tokenizer_file) -> None:
        tokenizer_file = 'Outputs/Tokenizers/{}'.format(tokenizer_file)
        tokenizer = Tokenizer.from_file('Outputs/Tokenizers/DistilBertTokenizerFastSW/DistilBertTokenizerFastSW')

        return tokenizer, tokenizer.get_vocab_size()

    def iteracoesDataset(self, train_dataset, test_dataset, inputs, tokenizer,c):
        return train_dataset, test_dataset

    def splitData(self, k, input_ids,tokenizer):

        # Instanciamos o StratifiedKFold
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        skf.get_n_splits(input_ids.sentence, input_ids.label)

        # Divide os dados em k_folds folds
        K_train_dataset = []
        K_test_dataset = []

        for train_index, test_index in skf.split(input_ids.sentence, input_ids.label):
            # Dividindo os dados em conjuntos de treinamento e validação
            K_train_dataset.append(input_ids.iloc[train_index])
            K_test_dataset.append(input_ids.iloc[test_index])

        return K_train_dataset, K_test_dataset

    def trainModel(self) -> None:
        pass

    def evalModel(self, model, test_inputs):

        top_pred_ids = model.predict(test_inputs)

        r_predict = []
        for item in top_pred_ids:
            r_predict.append(np.argmax(item))

        ground_truth_ids = []
        for element in test_inputs.as_numpy_iterator():
            aux = element[1]
            for item in aux:
                ground_truth_ids.append(item)

        taccuracy = accuracy_score(ground_truth_ids, r_predict)
        precision = precision_score(
            ground_truth_ids, r_predict, average='weighted')
        recall = recall_score(ground_truth_ids, r_predict, average='weighted')
        f1 = f1_score(ground_truth_ids, r_predict, average='weighted')
        wandb.log({
            "t-Accuracy": taccuracy,
            "t-Precision": precision,
            "t-Recall": recall,
            "t-F1-score": f1
        })

        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=ground_truth_ids, preds=r_predict,
                            class_names=self.le.classes_)})

    def logResults(self) -> None:
        pass


    def preprossing(self, inputs, tokenizer):
        le = preprocessing.LabelEncoder()
        data_labels_class = inputs.label.values
        le.fit(inputs.label.values)
        self.le = le
        data_labels = le.transform(inputs.label.values)
        inputs = inputs.assign(label=data_labels)

        return inputs, data_labels_class

    # These operations have to be implemented in subclasses.
    @abstractmethod
    def loadModel(self) -> None:
        pass

class modelDCNN(AbstractClass):
    """
    Concrete classes have to implement all abstract operations of the base
    class. They can also override some operations with a default implementation.
    """
    def loadModel(self, vocab_size, num_labels):

        Dcnn = DCNN(vocab_size=vocab_size,
                    emb_dim=128,
                    FFN_units=128,
                    nb_filters=50,
                    nb_classes=num_labels)

        Dcnn.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

        return Dcnn

    def trainModel(self, model, train_inputs, test_inputs, batch_size, epochs,hash_code):
        callbacks = []
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='loss', min_delta=0.01, patience=5))

        model.fit(
            train_inputs,
            validation_data=(test_inputs),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        model.save('Outputs/Models/ModelCheckpoints/{}'.format(hash_code),save_format='tf')
        return model

    def iteracoesDataset(self, train_dataset, test_dataset, inputs, tokenizer,c):
        u = Utils()
        iteracoesDataset = u.iteracoesDataset
        train_dataset, test_dataset = iteracoesDataset(train_dataset, test_dataset, inputs, tokenizer,c)

        return train_dataset, test_dataset

class modelDistilbert(AbstractClass):
    """
    Usually, concrete classes override only a fraction of base class'
    operations.
    """
    def _get_label_counts(self, label_list: List[str]) -> Counter:
        return Counter(label_list)

    def _get_valid_labels(self, label_counts: Counter) -> List[str]:
        valid_labels = []
        for label, count in label_counts.items():
            if count >= self.k:
                valid_labels.append(label)
        return valid_labels

    def _filter_dataset(self, valid_labels: List[str]) -> None:
        self.texts = [text for text, label in zip(
            self.texts, self.labels) if label in valid_labels]
        self.labels = [label for label in self.labels if label in valid_labels]

    def _prepare_dataset(self, texts: List[str], labels: List[str]):
        encoded = self.tokenizer.batch_encode_plus(texts, add_special_tokens=True,
                                                       max_length=50,
                                                       padding='max_length',
                                                       truncation=True,
                                                       return_attention_mask=True)
        return encoded
    def prepare(self, input_ids) -> List[List[int]]:
        label_counts = self._get_label_counts(input_ids.label)
        valid_labels = self._get_valid_labels(label_counts)
        self._filter_dataset(valid_labels)
        input_ids_list = self._prepare_dataset(self.texts, self.labels)
        return input_ids_list

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {'input_ids': inputs['input_ids'][0], 'attention_mask': inputs['attention_mask'][0], 'labels': label}


    def splitData(self, k, input_ids,tokenizer):
        self.k = k
        self.texts = input_ids.sentence
        self.labels = input_ids.label
        self.tokenizer = tokenizer
        input_ids_list = self.prepare(input_ids)
        # input_ids_list = pd.DataFrame(input_ids_list, columns=[
        #                               'input_ids', 'attention_mask'])

        # Cria uma lista de dicionários com as codificações de entrada e as classes correspondentes
        dataset = [{'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}
                   for input_ids, attention_mask, label in zip(input_ids_list['input_ids'],
                                                               input_ids_list['attention_mask'],
                                                               input_ids.label)]
        # Definimos o número de folds desejado
        k_folds = k

        # Instanciamos o StratifiedKFold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        # Divide os dados em k_folds folds
        train_datasets = []
        test_datasets = []
        for fold, (train_index, test_index) in enumerate(skf.split(dataset, input_ids.label)):
            # Separa os dados de treino e teste para cada fold
            train_datasets.append([dataset[i] for i in train_index])
            test_datasets.append([dataset[i] for i in test_index])

        return train_datasets, test_datasets

    def loadModel(self, vocab_size, num_labels):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels).to(device)
        model.resize_token_embeddings(vocab_size)
        return model

    def trainModel(self, model, train_inputs, test_inputs, batch_size, epochs,hash_code):

        learning_rate = 5e-5
        # Define os argumentos de treino
        args = TrainingArguments(output_dir='./Outputs/Models/ModelCheckpoints/{}'.format(hash_code),
                                evaluation_strategy='epoch',
                                learning_rate=learning_rate,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                num_train_epochs=epochs,
                                weight_decay=0.01,
                                push_to_hub=False,
                                load_best_model_at_end = True,
                                save_strategy='epoch',
                                save_total_limit = 5,
                                metric_for_best_model='eval_loss',
                                gradient_accumulation_steps = 2)

        trainer = Trainer(model=model,
                        args=args,
                        train_dataset=train_inputs,
                        eval_dataset=test_inputs,
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
                        )

        trainer.train()
        return trainer

    def loadTokenizer(self, tokenizer_file) -> None:
        added_tokens = ['hand', 'movement', 'dynamic', 'head', 'body', 'location', 'punctuation', 'estaticidadeRTrue', 'estaticidadeRFalse', 'estaticidadeLTrue', 'estaticidadeLFalse', 'hands_qtd0', 'hands_qtd1', 'hands_qtd2', 'right', 'left']
        tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name_or_path='Outputs/Tokenizers/DistilBertTokenizerFastSW', additional_special_tokens=added_tokens)

        return tokenizer, tokenizer.vocab_size

def client_code(abstract_class: AbstractClass, sweep_configuration) -> None:
    """
    O código do cliente chama o método de modelo para executar o algoritmo. O código cliente não precisa conhecer a classe concreta de um objeto com o qual trabalha, desde que trabalhe com objetos por meio da interface de sua classe base.
    """

    # ...
    abstract_class.template_method(sweep_configuration)
    # ...


if __name__ == "__main__":

    # Example sweep configuration
    parameters = {
        'dataset': ["details"],
        'language': ['asl'],
        'k': [8],
        'tokenizer': ['DistilBertTokenizerFastSW'],
        'model': ['Distilbert'],
        'batch_size':[16],
    }

# clf = client_code(modelDCNN(), sweep_configuration)

# Loop para percorrer todas as combinações de parâmetros
for config in product(*parameters.values()):

    client_code(modelDistilbert(), config)


# client_code(modelDistilbert(), sweep_id)
