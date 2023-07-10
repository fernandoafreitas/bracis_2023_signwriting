import exrex
import numpy as np

class Vocabulary:

    def __init__(self):
        self.batch_size = 1

    def especial_words(self):
        return ['hand', 'movement', 'dynamic', 'head', 'body', 'location', 'punctuation', 'estaticidadeRTrue', 'estaticidadeRFalse', 'estaticidadeLTrue', 'estaticidadeLFalse', 'hands_qtd0', 'hands_qtd1', 'hands_qtd2', '.', ' ', 'right', 'left',"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


    def generate_words(self):

        vocab = r'(s[123][0-9a-f]{2}[0-5][0-9a-f])'
        vocab1 = r'(s[123][0-9a-f][0-9a-f])'
        vocab2 = r'(s[123][0-9a-f][0-9a-f][0-9a-f])'
        all_words = list(exrex.generate(vocab))
        all_words2 = list(exrex.generate(vocab1))
        all_words3 = list(exrex.generate(vocab2))
        especial_words = self.especial_words()
        all_words = np.concatenate([especial_words,all_words, all_words2,all_words3])

        return all_words