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
START_SUF = "<<SUF_START>>"
START_PREF = "<<PREF_START>>"
END_SUF = "<<SUF_END>>"
END_PREF = "<<PREF_END>>"
MISS_SUF = "<<SUF_MISS>>"
MISS_PREF = "<<PREF_MISS>>"
PREF_LETTER = "<<PREF_LET>>"
SUF_LETTER = "<<SUF_LET>>"
# ---------------------------------------------


class DataContainer_ac():
    """
    This class provides with tools to use all needed data for training,validating and
    testing the window based tagger for both ner and pos tagging with a or c representations.
    The public members of the class contain all needed objects so that the user of the class
    could easily get all needed data and use.
    """
    def __init__(self,train_file,dev_file,test_file = None, pre_embd_words_path=None,
                 pre_embd_vectors_path=None ,use_sub_units= False, delim=" "):
        """
        The constructor initializes all public members(which are the data) according to passed parameters.
        Parameters could define 4 different options for initializing the data: with/without pre-trained vocab,
        with/without using sub units of the words.
        :param train_file: path to file with train data.
        :param dev_file: path to file with dev data.
        :param test_file: path to file with test data.
        :param pre_embd_words_path: path to file with words in pre vocab.
        :param pre_embd_vectors_path: path to file with vectors in pre vocab.
        :param use_sub_units: True if use sub units(prefixes and suffixes) in building data. False
        otherwise.
        :param delim: string which is a delimiter  between tags and input words in files with data set.
        """
        self.use_sub_units = use_sub_units
        self.pre_embd_vectors_path = pre_embd_vectors_path
        self.pre_embd_words_path = pre_embd_words_path
        self._test_file = test_file
        self._dev_file = dev_file
        self._train_file = train_file
        self.delim = delim
        self.max_sent_size = None
        self.test_x = None

        # create vocab and fill it with words from train set. Also, creates sentences from train set.
        self._create_vocab_words_and_senteces(train_file, delim)
        # if the path was provided - it means that user wants to use pre embedding.
        if pre_embd_vectors_path is not None:
            self._complete_vocab_words_from_pre_embedding()
        self._max_word_length = self._calculate_max_word_length()
        # create all mappings that link between words and indexes, laybels
        # and indexes and etc.
        self._create_indexes_mappings()
        dev_sentences = []
        if use_sub_units == True:
            self._create_prefix_suffix_dict(self.vocab)
            self._create_prefixes_to_indexes(len(self.vocab))
            self._create_suffixes_to_indexes(len(self.vocab) + len(self.prefix))

            self.dataset_train = self._create_set_with_sub_units(self.train_sentences)

            self.read_data(dev_file,dev_sentences,vocab=None, delim=delim)
            self.dataset_dev = self._create_set_with_sub_units(dev_sentences)

            if test_file is not None:
                test_sentences = []
                self.read_data(test_file,test_sentences,delim=delim,if_test=True)
                self.test_x = self._create_set_with_sub_units(test_sentences,if_tags=False)

        elif use_sub_units == False:

            self.dataset_train = self._create_set_without_sub_units(self.train_sentences)

            self.read_data(dev_file, dev_sentences, vocab=None, delim=delim)
            self.dataset_dev = self._create_set_without_sub_units(dev_sentences)

            if test_file is not None:
                test_sentences = None
                self.read_data(test_file,test_sentences,delim=delim,if_test=True)
                self.test_x = self._create_set_without_sub_units(test_file,if_tags=False)

        if pre_embd_words_path is None:
            self._create_embedding_without_pre_embedding()
        else:
            self._create_embedding_with_pre_embedding()

    def _calculate_max_word_length(self):
        all_words = [w for w in self.vocab]
        return len(max(all_words, key=len))

    def get_maximum_word_length(self):
        return self._max_word_length

    def complete_vocab(self, vocab):
        """
        This function completes vocab with 'special' words that handles specific cases in
        preparing the data for training. For example words that doesn't appear in vocab but could
        appear in dev or test sets.
        :param vocab- list of words(Strings) which is a vocab.:
        :return: the filled vocab(the same which was passed as parameter
        """
        vocab.add(END_WORD)
        vocab.add(MISS_WORD)
        return vocab

    def _create_vocab_words_and_senteces(self,train_file,delimiter):
        """
        This function creates vocabulary based on training set. Also it gathers sentences of words
        that appear in train file.
        :param train_file: path to train file.
        :param delimiter: seperator between tag and word in train set file
        :return: None
        """
        self.vocab = set()
        self.train_sentences = []
        self.complete_vocab(self.vocab)
        self.read_data(train_file, self.train_sentences, self.vocab, delim=delimiter)

    def _load_lines(self, file_path):
        """
        This is a help function to load all lines from file.
        :param file_path:  path to file.
        :return:  list, were each object in list is a string(line) withot '\n' sign at end.
        """
        lines = []
        with open(file_path) as f:
            lines = f.readlines()
        return [l.strip() for l in lines]

    def _complete_vocab_words_from_pre_embedding(self):
        """
        This function adds to main vocab(self.vocab) words that appear in pre trained embedding.
        :return: None
        """
        pre_embedding_words = self._load_lines(self.pre_embd_words_path)
        vocab_only_words = [pair[0] for pair in self.vocab]
        for pre_word in pre_embedding_words:
            if pre_word not in vocab_only_words:
                self.vocab.add(pre_word)

    def _create_indexes_mappings(self):
        """
        This function creates 2 members for the class:
        self.word_to_index, self.labels_to_index.

        each of them is a dictionary.
        words_to_index dictionary maps word to index of its vector
        in embedding matrix. labels_to_index maps from label to its index(each tag has its index
        in output of the network).
        :return:
        """
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        self.labels = set()
        for sent in self.train_sentences:
            for word,tag in sent:
                self.labels.add(tag)

        self.labels.add(MISS_TAG)
        self.labels_to_index = {l: i for l, i in zip(self.labels, range(len(self.labels)))}

    def _create_prefix_suffix_dict(self, vocab):
        """
        This function create suffix and prefix sets. That it is,
        it gets all prefixes and all suffixes from all worcs in the vocab
        and ads the to prefix and suffix dictionaries.
        :param vocab:
        :return:
        """
        self.prefix = set()
        self.suffix = set()
        for word in vocab:
            self.prefix.add(self._get_prefix(word))
            self.suffix.add(self._get_suffix(word))

    def _create_suffixes_to_indexes(self, start_indx):
        """
        This function creates mapping from suffix to index according to start index in embedding matrix.
        (We add suffixies to emdedding matrix so we need to put them somewhere and that's what these indices mean).
        All suffixes vectors in embedding will be one after another.
        :param start_indx: start index in embedding matrix for suffixes
        :return:
        """
        r = range(start_indx, len(self.suffix) + start_indx)
        self.suffixes_to_indexes = { a:i for a,i in zip(self.suffix, r)}

    def _create_prefixes_to_indexes(self, start_indx):
        """
        This function creates mapping from prefix to index according to start index in embedding matrix.
        (We add prefix to emdedding matrix so we need to put them somewhere and that's what these indices mean).
        All prefix vectors in embedding will be one after another.
        :param start_indx: start index in embedding matrix for prefix
        :return: None
        """
        r = range(start_indx, len(self.prefix) + start_indx)
        self.prefixes_to_indexes = { a:i for a,i in zip(self.prefix ,r)}

    def _create_set_with_sub_units(self,sentences,if_tags=True):
        """
        This function creates set using sub units and it supposes that sub units and other loaded data were handled with methods
        that add prefixes and suffixes to the embedding and that they have their own vectors.
        :param sentences - list with lists with words(strings) which represents list of sentences.
                Dataset will be created based on these sentences.

        :return: data set based on passed as parameter sentences. It's a list of tupples(x,y),
                where x is a one dim tensor of indexes of embedding vectors corresponding to sentence
                word, and y is a one dim tensor of indexes if labels corresponding to each sentence word,prefix,sufix,
                so that x[0] and y[0] correspond to the first word in a sentece, x[1] and y[2] to the second
                word and etc. In x, previously we store all words indexes in order, then all prefix
                indexes in order and then all sufix indexes in order so that all values are in the same
                tensor. In y, there only indexes of tags for each word(not sufixes/prefixes)
        """
        dataset = []
        inputs_words, inputs_prefixes, inputs_sufixes = self.structure_input_with_sub_units(sentences)
        for i in range(len(inputs_words)):
            # this is an 'x' of one tupple of dataset.
            input_t = torch.tensor(inputs_words[i][0] + inputs_prefixes[i] + inputs_sufixes[i])
            if if_tags:
                tags = torch.tensor(inputs_words[i][1])
                dataset.append((input_t, tags))
            else:
                dataset.append(input_t)

        return dataset

    def _create_set_without_sub_units(self,sentences,if_tags=True):
        """
        This function creates set without sub units(only words) and supposes that all other needed data were handled
        (creating vocab, indexes mapping, set of tags)
        :param sentences - list with lists with words(strings) which represents list of sentences.
                Dataset will be created based on these sentences.

        :return: data set based on passed as parameter sentences. It's a list of tupples(x,y),
                where x is a one dim tensor of indexes of embedding vectors corresponding to sentence
                word, and y is a one dim tensor of indexes if labels corresponding to each sentence word
                so that x[0] and y[0] correspond to the first word in a sentece, x[1] and y[2] to the second
                word and etc.
        """
        dataset = []
        inputs_words = self.structure_input_without_sub_units(sentences, test=if_tags)
        for i in range(len(inputs_words)):
            input_t = torch.tensor(inputs_words[i][0])
            if if_tags:
                tags = torch.tensor(inputs_words[i][1])
                dataset.append((input_t, tags))
            # for each word in sentece we append its tag to the data_train set
            else:
                dataset.append(input_t)

        return dataset

    def structure_input_with_sub_units(self, sentences):
        """
            This function takes as input the sentences of the data,
            replace each word that doesn't appear in the vocab by word "Miss",
            and get the prefix and the suffix of each word.
            its output is aword/prefix/suffix encoded sentence in their corresponding indexes
            :param sentences: sentences from which to structure input.
            :return:  inputs_words_tags(list of lists of tupples(sentece_words_indexes,
            sentece tags indexes), inputs_prefixes(list of of list with prefixes),
             inputs_sufixes(list of list with sufixes)
        """
        inputs_words_tags = []
        inputs_prefixes = []
        inputs_sufixes = []
        for sent in sentences:
            words_indexes = []
            tags_indexes = []
            pref_indexes = []
            suf_indexes = []

            for w,t in sent:
                if w not in self.word_to_index.keys():
                    w = MISS_WORD
                    t = MISS_TAG

                words_indexes.append( int(self.word_to_index[w]) )
                tags_indexes.append(int(self.labels_to_index[t]))
                pref_indexes.append(int(self.prefixes_to_indexes[self._get_prefix(w)]))
                suf_indexes.append(int(self.suffixes_to_indexes[self._get_suffix(w)]))

            inputs_words_tags.append((words_indexes,tags_indexes))
            inputs_prefixes.append(pref_indexes)
            inputs_sufixes.append(suf_indexes)

        return inputs_words_tags, inputs_prefixes, inputs_sufixes

    def structure_input_without_sub_units(self,sentences,test=False):
        """
        This function structures inputs to the network from given sentences. That is,
        this function passes through each sentence and gets all sentence words(indexes of their vectors
        in embedding matrix) and also gets the labels for them(tags of all words of
        five words) and puts it to list as tupple(five_word's_indexes, label) and returns this list.
        It doesn't handles sufixes and prefixes of the word.
        :param sentences: list with list of words(sentences) from which the input will be constructed.
        :param word_to_index: mapping from word to its index in embedding matrix.
        :param test: was not used.
        :return: list with tupples (vectors_of_five_words_indixes,label) which is a list with inputs to the network.
        """
        input_sequence = []
        for sent in sentences:
            words_indexes = []
            tags_indexes = []

            for word,t in sent:
                if word not in self.word_to_index.keys():
                    word = MISS_WORD
                if t not in self.labels_to_index.keys():
                    t = MISS_TAG
                words_indexes.append( int(self.word_to_index[word]) )
                tags_indexes.append( int(self.labels_to_index[t]) )

            input_sequence.append((words_indexes,tags_indexes))
        return input_sequence



    def read_data(self,data_file, sentences, vocab=None, delim=" ",if_test = False):
        """
            Sentences will contains all the sentences of the data file,
            and vocab (if was passed,that is if paramter vocab is
            different from None).
        :param data_file: file to read from
        :param sentences: list with senteces to fill from file.
        :param vocab: list to fill with word if passed(if it's not None).
        """
        sent = []
        with open(data_file) as f:
            for line in f:
                if line == "\n":
                    sentences.append(sent)
                    sent = []
                else:
                    if not if_test:
                        word, tag = line.strip().split(delim)
                    else:
                        word = line.strip()
                        tag = MISS_TAG
                    word = word.lower()
                    if vocab is not None:
                        vocab.add(word)
                        # vocab get each word and its label
                    sent.append((word,tag))
        return sentences, vocab

    def _get_prefix(self, word):
        """
        This function gets preffix from the word.
        If the word's length is less than 3(shortest prefix length in this program)
        then we add START_PREF char accordingly so that the length would still be 3.
        :param word: word to take prefix from.
        :return: preffix fo a word.
        """
        if word is START_WORD:
            return START_PREF
        elif word is END_WORD:
            return END_PREF
        elif word is MISS_WORD:
            return MISS_PREF
        elif len(word) >= 3:
            return word[0:3]
        if len(word) == 2:
            return PREF_LETTER + word
        else:
            return PREF_LETTER + PREF_LETTER + word

    def _get_suffix(self, word):
        """
             This function gets suffix from the word.
             If the word's length is less than 3(shortest prefix length in this program)
             then we add START_SUF char accordingly so that the length would still be 3.
             :param word: word to take sufix from.
             :return: sufix fo a word.
             """
        if word is START_WORD:
            return START_SUF
        elif word is END_WORD:
            return END_SUF
        elif word is MISS_WORD:
            return MISS_SUF
        elif len(word) >= 3:
            return word[-3:]
        if len(word) == 2:
            return word + SUF_LETTER
        else:
            return word + SUF_LETTER + SUF_LETTER


    def get_max_sentence_size(self):

        if self.max_sent_size is None:
            self.max_sent_size = len(max(self.train_sentences, key=len))
        return self.max_sent_size

    def _complete_sentence(self,sentence,difference):
        for i in range(difference):
            sentence.append(END_WORD)
        return sentence

    def _create_embedding_without_pre_embedding(self):
        """
        this function creates the embedding matrix that will feed the net,
              it is dimension are fitting the words_vocabulary, the prefix_vocab and the suffix_vocab sizes(if
              the use_sub_units is true).
              it is randomly initialize with the uniform[-1,1] Distribution.
              the embedding matrix will be initialized in member sef.Embedding_matrix
          """
        if self.use_sub_units:
            self.Embedding_matrix = nn.Embedding(len(self.vocab) + len(self.prefix) + len(self.suffix), EMBEDING_LEN)
        else:
            self.Embedding_matrix = nn.Embedding(len(self.vocab), EMBEDING_LEN)
        self.Embedding_matrix.weight.data.uniform_(-1, 1)

    def _create_embedding_with_pre_embedding(self):
        """
        This function initializes embedding matrix,where each word that appears in pre embedding
        gets pre trained vector, and each word that doesn't appear in pre embedding gets randomly initialized
        vector.
        the embedding matrix will be initialized in member sef.Embedding_matrix.
        :return: None.
        """
        vectors_from_pre_embedding = np.loadtxt(self.pre_embd_vectors_path)
        vectors_from_pre_embedding = torch.from_numpy(vectors_from_pre_embedding)
        #We before
        if self.use_sub_units:
            self.Embedding_matrix = nn.Embedding(len(self.vocab) + len(self.prefix) + len(self.suffix), EMBEDING_LEN)
        else:
            self.Embedding_matrix = nn.Embedding(len(self.vocab), EMBEDING_LEN)
        self.Embedding_matrix.weight.data.uniform_(-1, 1)
        with open(self.pre_embd_words_path) as f:
            words_from_pre_embedding = f.readlines()
        words_from_pre_embedding = [x.strip() for x in words_from_pre_embedding]
        # because word to index is bi directional, we easily can rotate it.
        index_to_word = {indx: word for word, indx in self.word_to_index.items()}
        if self.use_sub_units:
            for pref, indx in self.prefixes_to_indexes.items():
                index_to_word[indx] = pref
            for suf,indx in self.suffixes_to_indexes.items():
                index_to_word[indx] = suf
        # iterating over each word and its index from the main vocab.
        # if it will appear in pre embedding, we will take its pre vector
        # and to put it instead of randomly initialized vector.
        for index_vocab, word_vocab in index_to_word.items():
            if word_vocab in words_from_pre_embedding:
                index_in_pre_embedding = words_from_pre_embedding.index(word_vocab)
                pre_vector = vectors_from_pre_embedding[index_in_pre_embedding,:].clone()
                self.Embedding_matrix.weight.data[index_vocab,:] = pre_vector
        self.Embedding_matrix = self.Embedding_matrix.float()
        self.Embedding_matrix.weight.requires_grad = True

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

    def replace_test_set(self,test_file):
        """
        This function replaces test_set with another file.
        :param test_file:  file with test set to replace set.
        :return:
        """
        self._test_file = test_file
        sentences = []
        self.read_data(test_file,sentences,delim=self.delim,if_test=True)
        if self.use_sub_units:
            self.test_x = self._create_set_with_sub_units(sentences,if_tags=False)
        else:
            self.test_x = self._create_set_without_sub_units(sentences,if_tags=False)

    def move_sets_to_cpu(self):
        """
        This function moves train,dev and test sets to cpu.
        :return:
        """
        self.dataset_train = [(x.cpu(), y.cpu()) for x,y in self.dataset_train]
        self.dataset_dev = [(x.cpu(),y.cpu()) for x,y in self.dataset_dev]
        if self.test_x is not None:
            self.test_x = [x.cpu() for x in self.test_x]