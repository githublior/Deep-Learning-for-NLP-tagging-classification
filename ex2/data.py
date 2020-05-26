# Students: Lior Shimon, id: 341348498
#           Lev Levin, id: 342480456
import torch
from torch import nn
import numpy as np

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


class DataContainer():
    """
    This class provides with tools to use all needed data for training,validating and
    testing the window based tagger for both ner and pos tagging
    The public members of the class contain all needed objects so that the user of the class
    could easily get all needed data and use.
    """
    def __init__(self,train_file,dev_file,test_file, pre_embd_words_path=None,
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
        # create vocab and fill it with words from train set. Also, creates sentences from train set.
        self._create_vocab_words_and_senteces(train_file, delim)
        # if the path was provided - it means that user wants to use pre embedding.
        if pre_embd_vectors_path is not None:
            self._complete_vocab_words_from_pre_embedding()
        #create all mappings that link between words, labels, and vectors in embedding matrix.
        self._create_indexes_mappings_for_words()

        if use_sub_units == True:
            self._create_prefix_suffix_dict(self.vocab)
            self._create_prefixes_to_indexes(len(self.vocab))
            self._create_suffixes_to_indexes(len(self.vocab) + len(self.prefix))
            self._create_train_with_sub_units()
            self._create_dev_with_sub_units(dev_file,delim)

            self.create_test_with_sub_units(test_file)

        elif use_sub_units == False:
            self._create_train_without_sub_units()
            self._create_dev_without_sub_units(dev_file, delim)
            self._create_test_without_sub_units(test_file)

        if pre_embd_words_path is None:
            self._create_embedding_without_pre_embedding()
        else:
            self._create_embedding_with_pre_embedding()

    def complete_vocab(self, vocab):
        """
        This function completes vocab with 'special' words that handles specific cases in
        preparing the data for training. For example words that doesn't appear in vocab but could
        appear in dev or test sets.
        :param vocab- list of tupples (word,tag) which is a vocab.:
        :return:
        """
        vocab.add((START_WORD, MISS_TAG))
        vocab.add((END_WORD, MISS_TAG))
        vocab.add((MISS_WORD, MISS_TAG))
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
        self.sentences = []
        self.complete_vocab(self.vocab)
        self.read_data(train_file, self.sentences, self.vocab, delim=delimiter)

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
        Because self.vocab contains pairs: <word,tag> and it embedding there may be words that are
        not in train set and we don't know their tags, we will add special tag to such words MISS_TAG.
        :return: None
        """
        pre_embedding_words = self._load_lines(self.pre_embd_words_path)
        vocab_only_words = [pair[0] for pair in self.vocab]
        for pre_word in pre_embedding_words:
            if pre_word not in vocab_only_words:
                self.vocab.add((pre_word, MISS_TAG))

    def _create_indexes_mappings_for_words(self):
        """
        This function creates 3 members for the class:
        self.word_to_index, self.labels_to_index, self_Wx_to_Lx.
        each of them is a dictionary.
        words_to_index dictionary maps word to index of its vector
        in embedding matrix. labels_to_index maps from label to its index(each tag has its index
        in output of the network). self._Wx_to_Lx is a mapping from word index(index of vector in
        embedding) to index of tag.
        :return:
        """
        self.word_to_index = {w[0]: i for w, i in zip(self.vocab, range(len(self.vocab)))}
        self.labels = set()
        for pair in self.vocab:
            self.labels.add(pair[1])
        self.labels_to_index = {l: i for l, i in zip(self.labels, range(len(self.labels)))}

        self.Wx_to_Lx = {}
        for line in self.vocab:
            self.Wx_to_Lx[self.word_to_index[line[0]]] = self.labels_to_index[line[1]]

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
            self.prefix.add(self._get_prefix(word[0]))
            self.suffix.add(self._get_suffix(word[0]))

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

    def _create_train_with_sub_units(self):
        """
        This function creates train set and it supposes that all loaded data was handled with methods
        that add prefixes and suffixes to the embedding and that they have their own vectors.
        Train set - list with tupple  where first element is a vector of indicies of words in embedding(
        including sufixes and prefixes) and second ellement is an index of correct tag.

        :return:
        """
        self.dataset_train = []
        fives_words, fives_prefixes, fives_sufixes = self.structure_input_with_sub_units(self.sentences, self.word_to_index)
        for i in range(len(fives_words)):
            five_t = torch.tensor(fives_words[i] + fives_prefixes[i] + fives_sufixes[i])
            self.dataset_train.append((five_t, self.Wx_to_Lx[fives_words[i][2]]))

    def _create_dev_with_sub_units(self, dev_file, delimiter):
        """
            This function loads the validation set from the Dev_file,
            and structure the dev set as input of the net as self.dataset_dev, and it supposes
            that there all needed components for adding sufixes and prefixes(sufix set,prefix set and etc.)
        :param dev_file: path to dev file
        :return: None
        """

        dev_sent = []
        # Structuring the Dev Set
        dev_sent = []
        self.read_data(dev_file, dev_sent, delim=delimiter)
        self.dataset_dev = []
        fives_words, fives_prefixes, fives_sufixes = self.structure_input_with_sub_units(dev_sent, self.word_to_index)
        for i in range(len(fives_words)):
            five_t = torch.tensor(fives_words[i] + fives_prefixes[i] + fives_sufixes[i])
            self.dataset_dev.append((five_t, self.Wx_to_Lx[fives_words[i][2]]))

    def create_test_with_sub_units(self, test_file):
        """
            This function load the test set from the test_file,
            and structure the test set as input of the net as self.test_x.
            It supposes that the needed members for sufixes and prefixes handling
            already has been initialized(sufix set, prefix set, sufix to indexes. prefix to indexes)
            :param dev_file: test_file
        """
        sentences = []
        self.read_data(test_file,sentences)
        fives_words, fives_prefixes, fives_sufixes = self.structure_input_with_sub_units(
            sentences,self.word_to_index,test=True)
        self.test_x = []
        for i in range(len(fives_words)):
            self.test_x.append(torch.tensor(fives_words[i] + fives_prefixes[i] + fives_sufixes[i]))


    def structure_input_with_sub_units(self,sentences, word_to_index,test = False):
        """
            This function take as input the sentences of the data,
            add to each of them a Start sign , and a End sign,
            replace each word that doesn't appear in the vocab by word "Miss",
            and get the prefix and the suffix of each word.
            its output is a 5-word/prefix/suffix window encoded in their corresponding indexes
            :param dev_file: sentences from which to structure input.
            :param word_to_index: mapping from words to index.
        """
        fives_words = []
        fives_prefixes = []
        fives_sufixes = []
        for s in sentences:
            s.insert(0, START_WORD)
            s.insert(0, START_WORD)
            s.append(END_WORD)
            s.append(END_WORD)

            for i in range(2, len(s) - 2):
                five = s[i - 2:i + 3]
                temp_five_words = []
                temp_five_prefix = []
                temp_five_suffix = []
                for w in five:
                    if w not in word_to_index.keys():
                        w = MISS_WORD

                    temp_five_words.append(int(word_to_index[w]))
                    temp_five_prefix.append(int(self.prefixes_to_indexes[self._get_prefix(w)]))
                    temp_five_suffix.append(int(self.suffixes_to_indexes[self._get_suffix(w)]))

                fives_words.append(temp_five_words)
                fives_prefixes.append(temp_five_prefix)
                fives_sufixes.append(temp_five_suffix)
        return fives_words, fives_prefixes, fives_sufixes


    def read_data(self,data_file, sentences, vocab=None, delim=" "):
        """
            Sentences will contains all the sentences of the data file,
            and vocab get each word and their label(if was passed,that is if paramter vocab is
            different from None).
        :param data_file: file to read from
        :param sentences: list with senteces to fill from file.
        :param vocab: list to fill with tupples (word,tag) if passed(if it's not None).
        """
        sent = []
        with open(data_file) as f:
            for line in f:
                if line == "\n":
                    sentences.append(sent)
                    sent = []
                else:
                    splited = line.strip().split(delim)
                    splited[0] = splited[0].lower()
                    if vocab is not None:
                        vocab.add((splited[0], splited[1]))
                        # vocab get each word and its label
                    sent.append(splited[0])
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


    def _create_train_without_sub_units(self):
        """
        This function creates dataset_train which is a list of training examples
        without sufixes and prefixes. Each
        example is a tupple (tensor_of_input_vector, indx_of_correct tag.)
        :return:
        """
        self.dataset_train = []
        fives = self.structure_input_without_sub_units(self.sentences, self.word_to_index)
        for five in fives:
            five_t = torch.tensor(five)
            self.dataset_train.append((five_t, self.Wx_to_Lx[five[2]]))

    def _create_dev_without_sub_units(self,dev_file, delim):
        """
               This funnction create dev set without sub units(without Sufixes prefixes)
               :return:
               """
        dev_sent = []
        self.read_data(dev_file, dev_sent,delim=delim)
        self.dataset_dev = []
        fives = self.structure_input_without_sub_units(dev_sent, self.word_to_index)
        for five in fives:
            five_t = torch.tensor(five)
            self.dataset_dev.append((five_t, self.Wx_to_Lx[five[2]]))

    def _create_test_without_sub_units(self, test_file):
        """
        This function creates test_set(initializes self.test_x)
        for model that uses only words(not sufexes and not prefixes).
        :param test_file: path to file with test examples.
        :return:None.
        """
        sentences = []
        self.read_data(test_file,sentences)

        fives = self.structure_input_without_sub_units(sentences,self.word_to_index,test=True)
        self.test_x = [torch.tensor(five) for five in fives]

    def structure_input_without_sub_units(self,sentences, word_to_index,test=False):
        """
        This function structures inputs to the network from given sentences. That is,
        this function passes through each sentence and gets 5 words(indexes of their vectors
        in embedding matrix) and also gets the label for them(which is a tag of middle word of
        five words) and puts it to list as tupple(five_word's_indexes, label) and returns this list.
        It doesn't handles sufixes and prefixes of the word.
        :param sentences: list with list of words(sentences) from which the input will be constructed.
        :param word_to_index: mapping from word to its index in embedding matrix.
        :param test: was not used.
        :return: list with tupples (vectors_of_five_words_indixes,label) which is a list with inputs to the network.
        """
        fives = []
        for s in sentences:
            s.insert(0, START_WORD)
            s.insert(0, START_WORD)
            s.append(END_WORD)
            s.append(END_WORD)
            for i in range(2, len(s) - 2):
                five = s[i - 2:i + 3]
                temp = []
                for w in five:
                    # if test and len(temp) == 2: self.word_test_indx_to_word[five_number] = w
                    if w not in word_to_index.keys():
                        w = MISS_WORD

                    temp.append(int(word_to_index[w]))
                fives.append(temp)
        return fives

    def _create_embedding_without_pre_embedding(self):
        """
        this function creates the embedding matrix that will feed the net,
              it is dimension are fitting the words_vocabulary, the prefix_vocab and the suffix_vocab sizes(if
              the use_sub_units is true).
              it is randomly initialize with the uniform[-1,1] Distribution.
              the embedding matrix will be initialized in member sef.Embedding_matrix
          """
        if self.use_sub_units:
            self.Embedding_matrix = nn.Embedding(len(self.vocab) + len(self.prefix) + len(self.suffix), 50)
        else:
            self.Embedding_matrix = nn.Embedding(len(self.vocab), 50)
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
            self.Embedding_matrix = nn.Embedding(len(self.vocab) + len(self.prefix) + len(self.suffix), 50)
        else:
            self.Embedding_matrix = nn.Embedding(len(self.vocab), 50)
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






