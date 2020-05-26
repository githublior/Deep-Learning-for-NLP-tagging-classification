# Students: Lior Shimon, id: 341348498
#           Lev Levin, id: 342480456
import torch
import numpy as np
import torch.nn as nn
import random
import logging
from sys import argv
import pickle
from utils import *
import datetime




class Network(nn.Module):
    """
    The implementation of decomposable attention model for natural language inference.
    """
    def __init__(self, embedding, out_proj_size,out_lin_size, label_size):
        nn.Module.__init__(self)

        self.embedding = embedding
        self.out_project_size = out_proj_size
        self.label_size = label_size
        self.out_lin_size = out_lin_size

        # a projection layer.
        self.project_linear = nn.Linear(emb_dim, self.out_project_size)
        self.mlp_f = self._mlp_layers(self.out_project_size, self.out_lin_size)
        self.mlp_g = self._mlp_layers(2 * self.out_project_size, self.out_lin_size)
        self.mlp_h = self._mlp_layers(2 * self.out_lin_size,self.label_size)
        self._final_lin = nn.Linear(label_size,label_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)


    def _mlp_layers(self, input_dim, output_dim):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(0.2))
        mlp_layers.append(nn.Linear(input_dim, input_dim, bias=True))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(0.2))
        mlp_layers.append(nn.Linear(input_dim, output_dim, bias=True))
        # mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)

    def forward(self,sent1, sent2):

        len1= sent1.size(1)
        len2 = sent2.size(1)

        emb1 = self.embedding(sent1)
        emb2 = self.embedding(sent2)

        #size of s1: batch x len1 x out_proj_lin
        s1 = self.project_linear(emb1)
        # size of s2: batch x len2 x out_proj_lin
        s2 = self.project_linear(emb2)



        # ATTEND Step
        f1= self.mlp_f(s1.view(-1, self.out_project_size))
        f1 = f1.view(-1, len1, self.out_lin_size)
        # batch x len1 x out_lin_size
        f2 = self.mlp_f(s2.view(-1, self.out_project_size))
        f2= f2.view(-1, len2, self.out_lin_size)
        # batch x len2 x out_lin_size

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2))
        # e_{ij} batch_size x len1 x len2
        prob1 = torch.softmax(score1.view(-1, len2),dim=1).view(-1, len1, len2)
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        # e_{ji} batch_size x len2 x len1
        score2 = score2.contiguous()

        # batch_size x len2 x len1
        prob2 = torch.softmax(score2.view(-1, len1),dim=1).view(-1, len2, len1)


        # batch x len1 x out_proj_size
        beta = torch.bmm(prob1, s2)
        # batch X len2 x out_proj_size
        alpha = torch.bmm(prob2, s1)


        # COMPARE step
        # batch_size x len1 x (out_proj_size x 2)
        sent1_pre_g = torch.cat((s1, beta), 2)
        # batch_size x len2 x (out_proj_size x 2)
        sent2_pre_g = torch.cat((s2, alpha), 2)



        g1 = self.mlp_g(sent1_pre_g.view(-1, 2 * self.out_project_size))
        g2 = self.mlp_g(sent2_pre_g.view(-1, 2 * self.out_project_size))
        # batch_size x len1 x out_proj_size
        g1 = g1.view(-1, len1, self.out_project_size)
        # batch_size x len2 x out_proj_size
        g2 = g2.view(-1, len2, self.out_project_size)


        #AGGREGATE step
        v1 = torch.sum(g1, 1)  # batch_size x 1 x out_proj_size


        v2 = torch.sum(g2, 1)  # batch_size x 1 x out_proj_size


        v_concated = torch.cat((v1,v2),1) # batch x 2*out_proj_size
        h = self.mlp_h(v_concated)
        ret = self._final_lin(h)
        # final = nn.CrossEntropyLoss(h)  #TODO:check how this line is relevantl
        return ret


    def train_model(self, optimizer, criterion, data_train_loader, data_dev_loader, epochs, cuda, device, model_file ):
        """
        This function trains the model.
        :param optimizer: optimizer.
        :param criterion: loss function
        :param data_train_loader: loader with train data.
        :param data_dev_loader: loader with dev dat.
        :param epochs: num of epochs to train.
        :param cuda: boolean -True if cuda available, false otherwise.
        :param device: cpu or gpu.
        :return: list with vallidation accuracy per epoch , list with loss values per epoch.

        """

        #initializing the logfile:


        logging.info("Training started...")

        self.train()
        val_acc_list = []
        val_loss_list = []
        loss_train_list = []
        acc_train_list = []
        for e in range(epochs):
            t_start = datetime.datetime.now()
            print("epoch number :", e + 1)
            logging.info(f'epoch number:{e+1}')
            correct = 0
            total = 0
            losses_train_per_epoch_list = []
            for indx, batch in enumerate(data_train_loader):
                sent1 = batch.premise.T.clone()
                sent2 = batch.hypothesis.T.clone()
                label = batch.label.T.clone()

                sent1, sent2, label = sent1.to(device), sent2.to(device), label.to(device)
                # Run the forward pass
                outputs = self(sent1, sent2)
                new_label = label - 1
                loss = criterion(outputs, new_label)
                loss_train_list.append(loss.item())
                losses_train_per_epoch_list.append(loss.item())

                # Backprop and perform Adagrad optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy
                #outputs = self.softmax(outputs)
                _, predicted = torch.max(outputs, 1)
                total += predicted.shape[0]
                correct+= torch.sum(predicted == new_label).item()
            train_acc = 0
            if total != 0: train_acc = (correct / total) * 100


            train_loss = sum(losses_train_per_epoch_list) / len(losses_train_per_epoch_list)
            print('Epoch [{}/{}], Train_Loss: {:.2f}, Train_Accuracy: {:.2f}%'
                    .format(e + 1, epochs, train_loss, train_acc))

            dev_acc_loss = self.validate_model(data_dev_loader, cuda, device, criterion)
            print("Epoch [{}/{}] ,Dev_Loss:{:.2f},  Dev_Accuracy:{:.2f}%".format(e + 1, epochs,
                                                                                 dev_acc_loss[1], dev_acc_loss[0]))

            logging.info(
                'Epoch [{}/{}], Train_Loss: {:.3f}, Train_Accuracy: {:.2f}%, Dev Loss: {:.3f}, Dev_Accuracy: {:.2f}%'
                .format(e, epochs, train_loss, train_acc, dev_acc_loss[1], dev_acc_loss[0]))
            t_start = datetime.datetime.now()

            val_acc_list.append(dev_acc_loss[0] / 100)
            val_loss_list.append(dev_acc_loss[1])
            acc_train_list.append(train_acc)

            with open(TR_ACC_FILE,"wb+") as f:
                pickle.dump(acc_train_list,f)
            with open(DEV_ACC_FILE,"wb+") as f:
                pickle.dump(val_acc_list,f)
            logging.info("model,dev accuracies and train accuracies were saved to files.")
        logging.info("Training finished.")
        return val_acc_list, val_loss_list

    def validate_model(self, dev_set, cuda, device, criterion):
        """
        This function validates the model.
        :param dev_set: data loader with dev set.
        :param cuda: boolean - true if gpu is aviable, false otherwise.
        :param device: cpu or gpu.
        :param criterion: loss function
        :return: validation accuracy and validation loss.
        """
        loss_list = []
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in dev_set:
                sent1 = batch.premise.T.clone()
                sent2 = batch.hypothesis.T.clone()
                label = batch.label.T.clone()
                sent1 = sent1.to(device)
                sent2 = sent2.to(device)
                label = label.to(device)
                outputs = self(sent1, sent2)

                # because our labels are indexed from 0 to 2 and and not from 1 to 3.
                new_label = label - 1
                loss = criterion(outputs, new_label)
                loss_list.append(loss.item())
                # outputs = self.softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)
                total += predicted.shape[0]
                correct+= torch.sum(predicted == new_label).item()
            if total != 0:
                acc = correct * 100 / total
            else:
                print("total=0 !pbm")
                acc = 0
            self.train()
            # print(str(self.state_dict()))
            return acc, np.sum(loss_list)/len(loss_list)

    def test(self, test_input, cuda, device, criterion):
        """
        This function calculates accuracy on a given test set.
        :param test_input:
        :param cuda:
        :param device:
        :param criterion:
        :return:
        """
        a, b = self.validate_model(test_input, cuda, device, criterion)
        print(f"the final test accuracy is {a} , and the loss acc is {b}")
        logging.info(f"the final test accuracy is {a} , and the loss acc is {b}")
        return a,b