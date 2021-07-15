# Students: Lior Shimon, id: 341348498
#           Lev Levin, id: 342480456
import torch
from torch import nn
import numpy as np
from utils import *

# CONSTANTS------------------------------------
MISS_TAG = "111"
MISS_WORD = "<Miss>"
START_WORD = "<Start>"
END_WORD="<End>"
MISS_LET = "<Miss_w>"
START_LET = "<Start_l>"
END_LET="<End_l>"

#_-----------------------------------------------

class DataContainer_b():
    """
        This class provides with tools to use all needed data for training,validating and
        testing the window based tagger for both ner and pos tagging with b representations.
        The public members of the class contain all needed objects so that the user of the class
        could easily get all needed data and use.
        """
    def __init__(self,train_file,dev_file,test_file = None, pre_embd_words_path=None,pre_embd_vectors_path=None , delim=" "):
        """
            The constructor initializes all public members(which are the data) according to passed parameters.
            :param train_file: path to file with train data.
            :param dev_file: path to file with dev data.
            :param test_file: path to file with test data.
            :param pre_embd_words_path: path to file with words in pre vocab.
            :param pre_embd_vectors_path: path to file with vectors in pre vocab.
            :param use_sub_units: True if use sub units(prefixes and suffixes) in building data. False
            otherwise.
            :param delim: string which is a delimiter  between tags and input words in files with data set.
            """
        self.test_x = None
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self._max_word_len = None
        self.delim = delim
        self._create_vocabs_and_sentences( train_file, delim)
        self._create_indexes_mappings()

        self.dataset_train = self._create_set(self.train_sentences)

        self.dev_sentences = []
        self.read_data(dev_file,self.dev_sentences,delim=delim)
        self.dataset_dev = self._create_set(self.dev_sentences)

        if test_file is not None:
            self.test_sentences = []
            self.read_data(test_file,self.test_sentences,delim=delim,if_test=True)
            self.dataset_test = self._create_set(self.test_sentences,if_tags=False)
        self._create_embedding()


    def _create_vocabs_and_sentences(self, train_file, delim):
        """
        Creates vocab based on letters and vocab based on words and loads train
        sentences.
        :param train_file:
        :param delim: delimiter between word and tag in train file.
        :return:
        """
        self.vocab_letters = set()
        self.vocab_words = set()
        self.train_sentences = []
        self.complete_vocab_letters(self.vocab_letters)
        self.read_data(train_file, self.train_sentences, vocab_words=self.vocab_words, vocab_letters=self.vocab_letters,delim=delim)
        self.labels = set()
        count = 0
        for sent in self.train_sentences:
            for _, tag in sent:
                count += 1
                self.labels.add(tag)
        self.labels.add(MISS_TAG)



    def complete_vocab_letters(self, vocab):
        """
        This function completes vocab with 'special' l that handles specific cases in
        preparing the data for training. For example words that doesn't appear in vocab but could
        appear in dev or test sets.
        :param vocab- list of word which is a vocab.:
        :return:
        """
        vocab.add(END_LET)
        vocab.add(MISS_LET)


    def read_data(self,data_file,sentences, vocab_words=None, vocab_letters=None, delim=" ",if_test = False):
        with open(data_file) as f:
            sent = []
            for line in f:
                if line == "\n":
                    sentences.append(sent)
                    sent = []
                    continue
                if not if_test:
                    word, tag = line.strip().split(delim)
                else:
                    word = line.strip()
                    tag = MISS_TAG
                word = word.lower()
                sent.append((word, tag))
                if vocab_words is not None:
                    vocab_words.add(word)
                if vocab_letters is not None:
                    for let in word: vocab_letters.add(let)

    def _create_indexes_mappings(self):
        self.letter_to_index = {l: i for i, l in enumerate(self.vocab_letters)}
        self.labels_to_index = {l: i for i, l in enumerate(self.labels)}


    def _create_set(self,sentences,if_tags=True):
        """
        this function creates set bases on sentences.
        :param sentences:  list with list of words,tags.
        :param if_tags: if put tags in set or not.
        :return: list with tupple(sentenece_as_tensors_of_char_indexes, tags_tensor)
        """
        dataset = []
        letters_tags_sentences = self.structure_input(sentences)
        for i in range(len(letters_tags_sentences)):

            inputs_t = torch.tensor(letters_tags_sentences[i][0])
            if if_tags:
                tags = torch.tensor(letters_tags_sentences[i][1])
                dataset.append((inputs_t, tags))
            else:
                dataset.append(inputs_t)

        return dataset


    def get_maximum_word_length(self):
        if self._max_word_len is None:
            words = [w for w in self.vocab_words]
            self._max_word_len = len(max(words, key=len))
        return self._max_word_len

    def _complete_word(self,word,difference):
        for i in range(difference):
            word += END_LET
        return word

    def structure_input(self, sentences):
        """
        this function structures input according to sentences
        :param sentences list of list with pairs(words,tags):
        :return: structured input(list of tupple(list_of_indexes_char, list_tags))
        """
        self.get_maximum_word_length()
        words = {w for sent in sentences for w,_ in sent}
        maximum_len = len(max(words, key=len))
        self._max_word_len = max(maximum_len, self._max_word_len)
        sentences_letters = []
        for sent in sentences:
            tags_indexes = []
            letters_in_one_sent = []
            for word,tag in sent:

                if tag not in self.labels_to_index.keys():
                    tag = MISS_TAG
                tags_indexes.append(self.labels_to_index[tag])


                #treating letters
                one_word_indexes = []
                for let in word:
                    if let in self.vocab_letters: one_word_indexes.append(self.letter_to_index[let])
                    else: one_word_indexes.append(self.letter_to_index[MISS_LET])
                # padding.
                if len(word) < self._max_word_len:
                    for i in range(self._max_word_len - len(word)): one_word_indexes.append(self.letter_to_index[END_LET])
                letters_in_one_sent.append(one_word_indexes)

            sentences_letters.append((letters_in_one_sent, tags_indexes))
        return sentences_letters

    def _create_embedding(self):
        """
        this function creates the embedding matrix that will feed the net,
              it is dimension are fitting the words_vocabulary
              it is randomly initialize with the uniform[-1,1] Distribution.
              the embedding matrix will be initialized in member sef.Embedding_matrix
          """

        self.Embedding_matrix = torch.nn.Embedding(len(self.vocab_letters),EMBEDING_LEN )
        self.Embedding_matrix.weight.data.uniform_(-1, 1)


    def replace_test_set(self,test_file):
        self._test_file = test_file
        sentences = []
        self.read_data(test_file,sentences,delim=self.delim,if_test=True)
        self.test_x = self._create_set(sentences,if_tags=False)

    def move_sets_to_cpu(self):
        self.dataset_train = [(x.cpu(), y.cpu()) for x,y in self.dataset_train]
        self.dataset_dev = [(x.cpu(),y.cpu()) for x,y in self.dataset_dev]
        if self.test_x is not None:
            self.test_x = [x.cpu() for x in self.test_x]

    def load_test_words(self):
        """
        This function loads test set. It has its own function because it contains only sentences
        without tags.
        :return: list with words in test set.
        """
        words = []
        with open(self._test_file) as f:
            for line in f:
                if line != "\n":
                    words.append(line.strip())

        return words
