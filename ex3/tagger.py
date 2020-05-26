import torch
import numpy as np
from utils import *
import torch.nn as nn
import random
from sys import argv
class network(nn.Module):
    """
    This class is an implementation of the BiLSTM Network.
    composed of 2 BiLSTM Layer, one hidden layer and softmax.
    """
    def __init__(self,Embedding_matrix, num_of_features_for_blstm, blstm_output_size,output_s,repr, max_word_len=None,if_cud = False,
                 device = 'cpu'):
        """

        :param Embedding_matrix: nn.embedding instance
        :param num_of_features_for_blstm: size of each input unit for blstm.
        :param blstm_output_size: size of each output unit of blstm.
        :param output_s: output size from linear layer.
        :param repr: sentence representation
        :param max_word_len: maximum word len in train set.
        :param if_cud: boolean saying if cuda is avialable.
        :param device: device to use on it model.
        """
        nn.Module.__init__(self)

        self.max_word_len = max_word_len
        self.embedding = Embedding_matrix

        self.if_cuda = if_cud
        self.device = device
        self.lstm = nn.LSTM(EMBEDING_LEN,num_of_features_for_blstm, batch_first=True)
        self.num_of_features = num_of_features_for_blstm
        self.bilstm = nn.LSTM(input_size=num_of_features_for_blstm,hidden_size=blstm_output_size, num_layers=2 ,bidirectional=True, batch_first=True,dropout=DROPOUT)

        self.linear = nn.Linear(blstm_output_size *2 ,output_s)
        if if_cud:
            x1 = torch.zeros(4, 1, blstm_output_size).cuda()
            x2 = torch.zeros(4,1,blstm_output_size).cuda()
        else:
            x1 = torch.zeros(4, 1, blstm_output_size)
            x2 = torch.zeros(4, 1, blstm_output_size)
        x1 = x1.to(device)
        x2 = x2.to(device)
        self.hidden_start_cell = (x1,x2)
        self.tanh = torch.nn.Tanh()
        # start hidden cell that we will use in computing representations of words according to its characters

        # we don't use it:
        # self.hidden_start_cell_lstm = (torch.zeros(1,max_sent_len,EMBEDING_LEN),torch.zeros(1,max_sent_len,EMBEDING_LEN))
        self.repr= repr

        self.get_blistm_input = None
        if repr == REPR_A:
            self.get_blistm_input = self.compute_a_repr
        elif repr == REPR_B:
            self.get_blistm_input = self.compute_b_repr
        elif repr == REPR_C:
            self.get_blistm_input = self.compute_c_repr
        elif repr == REPR_D:
            self.get_blistm_input = self.compute_d_repr
            self.prelin_d = nn.Linear(EMBEDING_LEN * 2, self.num_of_features)





    def forward(self, seq_indexes):
        '''

        :param seq_indexes are indexes for a sequence of shape (batch_size, num_of_elem_in_seq, len_of_each_elem_in_seq) or
            seq is sometime sentences(words) sometimes word(letters)

        :return:
        '''
        # fetches embedding vectors according to indexes and concates them.
        output = self.get_blistm_input(seq_indexes)
        output, _ = self.bilstm(output, self.hidden_start_cell)
        #removing this dimension in order to fit with the rest of the network
        output = output.squeeze(0)
        output = self.linear(output)
        return output

    def compute_a_repr(self,seq_indexes):
        """
        prepares input for bilstm.
        :param seq_indexes:
        :return:
        """
        return self.embedding(seq_indexes)

    def compute_b_repr(self,seq_indexes):
        """
        prepares input for bilstm.
        :param seq_indexes: tensor of shape (1,num_of_words,num_of_characters)

        :return: prepared input for blstm.
        """
        # concated_embeds will be of the shape (1,num_of_words,num_of_characters,len_of_char_embed_vector)
        concated_embeds = self.embedding(seq_indexes)
        concated_embeds.squeeze_(0)

        x1 = torch.zeros(1,concated_embeds.shape[0] , self.num_of_features)
        x2 = torch.zeros(1, concated_embeds.shape[0], self.num_of_features)
        if self.if_cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        hidden_start_cell_lstm = (x1,x2)
        # lstm output will be of the shape(num_of_words,num_of_characters,num_of_features). We take
        # the last output vector of lstm.
        lstm_output,_ = self.lstm(concated_embeds, hidden_start_cell_lstm)
        # result will be of shape(num of words, num_of_features)
        result = lstm_output[:,seq_indexes.shape[2] - 1,:].clone()
        result.squeeze_(1)
        return  self.tanh(result.unsqueeze(0))
    def compute_c_repr(self,seq_indexes):
        """
        prepares input for bilstm.
        :param seq_indexes:
        :return: prepated input for bilstm.
        """
        concated_embeds = self.embedding(seq_indexes)
        size_base = concated_embeds.shape[1] // 3
        concated_embeds = concated_embeds[:,0: size_base,:] + concated_embeds[:,size_base: size_base * 2,: ] +concated_embeds[:, size_base*2:,:]
        return concated_embeds
    def compute_d_repr(self,seq_indexes):
        """
        prepares input for bilstm.
        :param seq_indexes: tensor of shape(1,num_of_words,max_num_of_letters)
        :return:  prepared input for bilstm.
        """
        sentence_len = seq_indexes[0,-1].item()

        # words_embed_indx has shape (1, num_of_words)
        words_embed_indexes = seq_indexes[:,0:sentence_len]
        words_embed = self.embedding(words_embed_indexes)
        # letters_embed_indx has shape (1, num_of_words*max_num_of_letters)
        letters_embed_indexes = seq_indexes[:,sentence_len:-1]

        # concated_embeds_letters has shape (1,num_of_words*max_num_of_letters, Embedding_len)
        concated_embeds_letters = self.embedding(letters_embed_indexes)
        # concated_embeds_letters has shape (num_of_words*max_num_of_letters, Embedding_len)
        concated_embeds_letters.squeeze_(0)
        concated_embeds_letters = concated_embeds_letters.view((sentence_len,self.max_word_len, EMBEDING_LEN))
        x1 = torch.zeros(1, sentence_len, self.num_of_features)
        x2 = torch.zeros(1, sentence_len, self.num_of_features)
        if self.if_cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        hidden_start_cell_lstm = (x1,x2)

        lstm_output, _ = self.lstm(concated_embeds_letters, hidden_start_cell_lstm)
        # now we take the last output of lstm
        result_letter = lstm_output[:,self.max_word_len - 1,:].clone()
        result_letter.squeeze_(1)
        # ready v
        result_letter.unsqueeze_(0)

        result_letter =self.tanh(result_letter)
        #we concat results from 'a' and 'b' computations and feed it into linear layer,
        #then we use tanh activation on it. we use squeeze(0) because the linear layer
        #can get only two dimensional tensor and our 0s dim is 1 so we can remove it.
        concat_all = torch.cat((words_embed, result_letter), 2).squeeze(0)
        # we unsqueeze 0s dim because afterwards it will be passed to bilstm.
        return self.tanh(self.prelin_d(concat_all)).unsqueeze(0)






    def validate_model(self, dev_set, cuda, device, criterion, O_to_index=None,miss_index = None):
        """
        This function validates the model.
        :param dev_set: data loader with dev set.
        :param cuda: boolean - true if gpu is aviable, false otherwise.
        :param device: cpu or gpu.
        :param criterion: loss function
        :param O_to_index: index of O tag.
        :param miss_index: index if <Miss> tag.
        :return: validation accuracy and validation loss.
        """
        count = True
        loss_list = []
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in dev_set:
                label = labels.squeeze(0)
                if cuda:
                    data, label = data.cuda(), label.cuda()
                data = data.to(device)
                label = label.to(device)

                outputs = self(data)
                #print("outputs: " + str(outputs))
                #print("label:" + str(label))
                #print("\n\n\n")
                loss = criterion(outputs, label)
                loss_list.append(loss.item())
              #  outputs = self.softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)
                for i,(p,l) in enumerate(zip(predicted,label)):
                    # if we work with pos tags.
                    if (O_to_index is None) or not (p == O_to_index and l == O_to_index):
                        total += 1
                        # Here we check whether a predicted laybel is a laybel "miss", which we artificially
                        # created to indicate on words that don't belong to the vocab of train/embeddings.
                        # It allows us to train the network more precisiously because it can detect words which
                        # are not in the vocab and still predict on them. But because we add an artificial laybel
                        # that cannot be used as a laybel for checking accuracy and for producing predictions
                        # on test, we need to get a prediction on real laybels(part-of-speech) and this still
                        # can be done even if the network produces outputing laybel "miss" - we just instead of
                        # taking the laybel that got the highest propability after softmax, need to take the laybel
                        # that got the second highest propability. So this condition is an implementation of that.
                        if p == miss_index:
                            sorted_idicies = torch.argsort(outputs.data[i], descending=True)
                            correct += (sorted_idicies[1] == l).sum().item()
                        else:
                            correct += (p == l).sum().item()
            if total != 0:
                acc = correct * 100 / total
            else:
                acc = 0
            self.train()
            return acc, np.sum(loss_list) / len(loss_list)

    def train_model(self, optimizer, criterion, data_train_loader, data_dev_loader, epochs, cuda, device, O_to_index=None,
                    miss_index = None):
        """
        This function trains the model.
        :param optimizer: optimizer.
        :param criterion: loss function
        :param data_train_loader: loader with train data.
        :param data_dev_loader: loader with dev dat.
        :param epochs: num of epochs to train.
        :param cuda: boolean -True if cuda available, false otherwise.
        :param device: cpu or gpu.
        :param O_to_index: index of O tag.
        :param miss_index: index of <Miss> tag.
        :return: list with vallidation accuracy per epoch , list with loss values per epoch.
        """
        self.train()
        val_acc_per_epoch = []
        val_loss_per_epoch = []

        for e in range(epochs):
            print("epoch number :", e + 1)
            correct = 0
            total = 0
            loss_list = []
            for indx, (data, labels) in enumerate(data_train_loader):
                # if indx == 2000:
                #     return val_acc_per_epoch, val_loss_per_epoch
                label = labels.squeeze(0)
                if cuda:
                    data, label = data.cuda(), label.cuda()
                    data = data.to(device)
                    label = label.to(device)

                data, label = data.to(device), label.to(device)
                # Run the forward pass
                outputs = self(data)
                loss = criterion(outputs, label)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
          #      outputs = self.softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)
                for i,(p, l) in enumerate(zip(predicted,label)):
                    if (O_to_index is None) or not (p == O_to_index and l == O_to_index):
                        total += 1
                        # Here we check whether a predicted laybel is a laybel "miss", which we artificially
                        # created to indicate on words that don't belong to the vocab of train/embeddings.
                        # It allows us to train the network more precisiously because it can detect words which
                        # are not in the vocab and still predict on them. But because we add an artificial laybel
                        # that cannot be used as a laybel for checking accuracy and for producing predictions
                        # on test, we need to get a prediction on real laybels(part-of-speech) and this still
                        # can be done even if the network produces outputing laybel "miss" - we just instead of
                        # taking the laybel that got the highest propability after softmax, need to take the laybel
                        # that got the second highest propability. So this condition is an implementation of that.
                        if p == miss_index:
                            sorted_idicies = torch.argsort(outputs.data[i], descending=True)
                            correct += (sorted_idicies[1] == l).sum().item()
                        else:
                            correct += (p == l).sum().item()
                if (indx + 1) % (NUM_OF_SENTENCES_TO_PRINT) == 0:
                    train_acc = 0
                    if total != 0: train_acc = (correct / total) * 100

                    train_loss = sum(loss_list) / len(loss_list)
                    print('Sentence [{}], Epoch [{}/{}], Train_Loss: {:.4f}, Train_Accuracy: {:.2f}%'
                    .format(indx,e + 1, epochs, train_loss, train_acc))

                    dev_acc_loss = self.validate_model(data_dev_loader, cuda, device, criterion,O_to_index,miss_index)
                    print("Sentence [{}], Epoch [{}/{}] ,Dev_Loss:{:.4f},  Dev_Accuracy:{:.2f}%".format(indx,e + 1,epochs,
                                                                               dev_acc_loss[1],dev_acc_loss[0]))
                    val_acc_per_epoch.append(dev_acc_loss[0] / 100)
                    val_loss_per_epoch.append(dev_acc_loss[1])


        return val_acc_per_epoch, val_loss_per_epoch

    def test(self,test_inputs,file_to_write,data_container ,cuda, device,routing_type):
        """
        This function tests the model.
        :param test_inputs: list of tensors that are test inputs to the model.
        :param file_to_write: file to write predictions to
        :param data_container: data_container object that appropriates to this model.
        :param cuda: if cuda is avialable.
        :param device: device(cpu or gpu)
        :param routing_type: ner or pos
        :return:
        """

        self.eval()
        delim = DELIM_POS if routing_type == POS_ROUTINE else DELIM_NER
        miss_tag_indx = data_container.labels_to_index[MISS_TAG]
        #loading list with test sentences words.
        test_words = data_container.load_test_words()
        word_indx = 0
        with open(file_to_write, "w+") as file:
            for input in test_inputs:
                five_n = input.unsqueeze(0)
                if cuda:
                  five_n = five_n.cuda()
                five_n.to(device)
                outputs = self(five_n)
                _, predicted = torch.max(outputs.data, 1)
                for i in range(predicted.shape[0]):

                    indx = predicted[i].item()
                    # if we predicted for miss tag, which is not real tag, we will get the tag
                    # that got the second highest probability after applying softmax.
                    if indx is miss_tag_indx:
                        sorted_indicies = torch.argsort(outputs.data[i],descending=True)
                        indx = sorted_indicies[1]
                    for key, val in data_container.labels_to_index.items():
                        if val == indx:
                            file.write(test_words[word_indx] + delim + key + "\n")
                            word_indx+=1
                file.write("\n")