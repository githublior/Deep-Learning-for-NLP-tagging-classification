# Students: Lior Shimon, id: 341348498
#           Lev Levin, id: 342480456

import torch.nn as nn
import torch
import numpy as np
from data import *

class tagger(nn.Module):
    """
    This class is an implementation of window based tagger.
    """
    def __init__(self, Embedding_matrix, input_size, hidden_size, output_size):
        """
        The constructor.
        :param Embedding_matrix: ready embedding matrix which the tagger will continue to learn.
        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        nn.Module.__init__(self)
        self.embedding = Embedding_matrix
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_indexes):
        output = self.embedding_concat(x_indexes)
        output = self.lin1(output)
        output = self.tanh(output)
        output = self.lin2(output)
        output = self.softmax(output)
        return output

    def embedding_concat(self,x_index_vector):
        """
        This function gets embedding of words that are input(input it's x_index_vector
         which represented as indices of needed embedding vectors)) and concats them.
         Also, if there are 15 elements in input vector, it indicates that the network
         works with sub-units - that there also indices of sub units(5 words, 5 prefixes,
         5 sufixes and each of these fives will be added with others).
        :param x_index_vector: input vector with indices of vectors in embedding.
        :return: concated vector.
        """
        result = self.embedding(x_index_vector).view((x_index_vector.shape[0], -1))
        if x_index_vector.shape[1] == 15:
            result = result[:, 0:250] + result[:, 250:500] + result[:, 500:750]
        return result

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
            for data, label in dev_set:
                if cuda:
                    data, label = data.cuda(), label.cuda()
                data = data.to(device)
                label = label.to(device)

                outputs = self(data)
                loss = criterion(outputs, label)
                loss_list.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                for indx,(pr,lb) in enumerate(zip(predicted,label)):
                    # if we work with pos tags.
                    if (O_to_index is None):
                        total += 1
                        correct += (pr == lb).sum().item()
                        # if we work with ner tags, count correct prediction if it was not O tag.
                    elif not (lb == O_to_index and pr == O_to_index):
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
                        if pr == miss_index:
                            sorted_idicies = torch.argsort(outputs.data[indx], descending=True)
                            correct += (sorted_idicies[1] == lb).sum().item()
                        else:
                            correct += (pr == lb).sum().item()





            if total != 0:
                acc = correct * 100 / total
            else:
                acc = 0
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
        count = True
        val_acc_per_epoch = []
        val_loss_per_epoch = []

        local_loss = 0
        acc_list = []
        total_step = len(data_train_loader)

        for e in range(epochs):
            print("epoch number :", e + 1)
            correct = 0
            total = 0
            loss_list = []
            for indx, (data, label) in enumerate(data_train_loader):

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

                _, predicted = torch.max(outputs.data, 1)
                for i,(p, l) in enumerate(zip(predicted,label)):

                    if (O_to_index is None):
                        total += 1
                        correct += (p == l).sum().item()
                    elif not (p == O_to_index and l == O_to_index):
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
            train_acc = 0
            if total != 0: train_acc = (correct / total) * 100

            train_loss = sum(loss_list) / len(loss_list)
            print('Epoch [{}/{}], Train_Loss: {:.4f}, Train_Accuracy: {:.2f}%'
                  .format(e + 1, epochs, train_loss, train_acc))

            dev_acc_loss = self.validate_model(data_dev_loader, cuda, device, criterion,O_to_index,miss_index)
            print("Epoch [{}/{}] ,Dev_Loss:{:.4f},  Dev_Accuracy:{:.2f}%".format(e + 1,epochs,
                                                                               dev_acc_loss[1],dev_acc_loss[0]))
            val_acc_per_epoch.append(dev_acc_loss[0] / 100)
            val_loss_per_epoch.append(dev_acc_loss[1])
        return val_acc_per_epoch, val_loss_per_epoch


    def test_model(self,test_fives,file_to_write,data_container ,cuda, device):
        """
        This function tests the model.
        :param test_fives:  list with five words to test on them.
        :param file_to_write:  path to file were the predictions will be saved.
        :param data_container: object that contains all needed data.
        :param cuda: true if gpu is available, False otherwise.
        :param device: cpu or gpu etc.
        :return:None.
        """
        self.eval()
        miss_tag_indx = data_container.labels_to_index[MISS_TAG]
        #loading list with test five words.
        test_words = data_container.load_test_words()
        with open(file_to_write, "w+") as file:
            for word, five in zip(test_words,test_fives):
                five_n = torch.zeros((1,five.shape[0]),dtype=torch.int64)
                five_n[0,:] = five
                if cuda:
                  five_n = five_n.cuda()
                five_n.to(device)
                outputs = self(five_n)
                _, predicted = torch.max(outputs.data, 1)
                indx = predicted.item()
                # if we predicted for miss tag, which is not real tag, we will get the tag
                # that got the second highest probability after applying softmax.
                if indx is miss_tag_indx:
                    sorted_indicies = torch.argsort(outputs.data[0],descending=True)
                    indx = sorted_indicies[1]
                for key, val in data_container.labels_to_index.items():
                    if val == indx:
                        file.write(word + " " + key + "\n")