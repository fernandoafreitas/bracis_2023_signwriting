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
from sklearn.model_selection import StratifiedKFold
from Resources.DCNN import DCNN
import numpy as np
from Resources.Utils import Utils
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tokenizers import Tokenizer


class AbstractClass(ABC):

    def template_method(self, s):
        """
        O método template define o esqueleto de um algoritmo.
        """
        dataset, language, k, tokenizer, model,batch_size = s

        epochs = 200
        u = Utils()
        iteracoesDataset = u.iteracoesDataset

        inputs = self.loadDatabase(language, dataset, k)
        sanitized_inputs, labels = self.preprossing(inputs, tokenizer)
        train_inputs, test_inputs = self.splitData(k,sanitized_inputs)
        tokenizer = self.loadTokenizer(tokenizer)

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
            train_dataset, test_dataset = iteracoesDataset(train_dataset, test_dataset, inputs, tokenizer,c)
            VOCAB_SIZE = tokenizer.get_vocab_size()
            NB_CLASSES = num_labels

            model = self.loadModel(VOCAB_SIZE, NB_CLASSES)

            model = self.trainModel(model, train_dataset, test_dataset, batch_size, epochs,hash_code)

            self.evalModel(model, test_dataset)

            self.disconnectWandb()

    # These operations already have implementations.

    def conectaWandb(self, s, i,hash_code):
        load_dotenv()
        os.environ["WANDB_API_KEY"] = os.environ['WANDB_API_KEY']
        dataset, language, k, tokenizer, model,batch_size = s

        wandb.init(project="bracis", entity="fernandofreitasusp",
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

        return tokenizer

    def splitData(self, k, input_ids):
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

    def evalModel(self) -> None:
        pass

    def logResults(self) -> None:
        pass

    # These operations have to be implemented in subclasses.

    @abstractmethod
    def preprossing(self) -> None:
        pass

    @abstractmethod
    def loadModel(self) -> None:
        pass

    # These are "hooks." Subclasses may override them, but it's not mandatory
    # since the hooks already have default (but empty) implementation. Hooks
    # provide additional extension points in some crucial places of the
    # algorithm.

    # def hook1(self) -> None:
    #     pass

    # def hook2(self) -> None:
    #     pass


class modelDCNN(AbstractClass):
    """
    Concrete classes have to implement all abstract operations of the base
    class. They can also override some operations with a default implementation.
    """

    def preprossing(self, inputs, tokenizer) -> None:
        le = preprocessing.LabelEncoder()
        data_labels_class = inputs.label.values
        le.fit(inputs.label.values)
        self.le = le
        data_labels = le.transform(inputs.label.values)
        inputs = inputs.assign(label=data_labels)


        # Tokenizando as frases de entrada
        # max_length = 512
        # tokens = tokenizer.batch_encode_plus(
        #     inputs['sentence'].tolist(),
        #     truncation=True,
        #     padding='max_length',
        #     max_length=max_length,
        #     return_token_type_ids=False
        # )

        # Convertendo as sequências de tokens para arrays numpy
        # input_ids = pad_sequences(
        #     tokens['input_ids'], maxlen=max_length, dtype='long', truncating='post', padding='post')

        # labels = inputs['label'].astype('category').cat.codes

        # token_lengths = [len(tokens) for tokens in tokens['input_ids']]
        # max_token_length = max(token_lengths)
        # print(f'Max token length: {max_token_length}')

        return inputs, data_labels_class

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



class modelDistilbert(AbstractClass):
    """
    Usually, concrete classes override only a fraction of base class'
    operations.
    """

    def preprossing(self, inputs) -> None:
        print("AbstractClass says: But I am doing the bulk of the work anyway")

    # def loadModel(self, num_labels, tokenizer) -> None:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = DistilBertForSequenceClassification.from_pretrained(
    #         model_name, num_labels=num_labels).to(device)
    #     model.resize_token_embeddings(len(tokenizer))
    #     return model


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
        'language': ['libras'],
        'k': [8],
        'tokenizer': ['DistilBertTokenizerFastSW'],
        'model': ['DCNN'],
        'batch_size':[16],
    }

# clf = client_code(modelDCNN(), sweep_configuration)

# Loop para percorrer todas as combinações de parâmetros
for config in product(*parameters.values()):

    client_code(modelDCNN(), config)


# client_code(modelDistilbert(), sweep_id)
