import numpy as np
# from gen_examples import MAX_RANDOM as MAX_SEQ_SIZE
from sys import argv
import torch.utils.data.dataloader as dl
import torch
import torch.nn as nn
LETTER_TO_INDEX = {str(i):i for i in range(1,10)}
LETTER_TO_INDEX['a'] = 10
LETTER_TO_INDEX['b'] = 11
LETTER_TO_INDEX['c'] = 12
LETTER_TO_INDEX['d'] = 13

TAG_INDX_NEG = 0
TAG_INDX_POS = 1
POS_FILE = 'pos_examples'
NEG_FILE = 'neg_examples'
MAX_SEQ_SIZE =1000


def read_data(file_name):
    """
    Reads examples of data from file and returns list of strings,
    each string is an example, without a tag.
    :param file_name:
    :return:
    """

    data = []
    with open(file_name) as f:
        for line in f:
            seq = line.strip()
            data.append(seq)
    return data


def encode_data(data):
    """
    encode each symbol of each sentence

    :param data: input_file as al list of list
    :return: list of tensors of encoded data
    """
    encoded_d = []
    for ex in data:
        counter = 0
        example = [LETTER_TO_INDEX[let] for let in ex]
        encoded_d.append(torch.FloatTensor(example))
    return encoded_d

def resized_encoded_data(encoded_d):
    """
    resized the data so that each sentence has the same size. fill the gaps withs 0's.
    """
    labels = [x[1] for x in encoded_d]
    inputs = [x[0] for x in encoded_d]
    inputs.append(torch.zeros(MAX_SEQ_SIZE))
    resized_x = torch.nn.utils.rnn.pad_sequence(inputs, batch_first =True)
    resized_x = resized_x[0:-1,:]
    final = [(resized_x[i,:],labels[i]) for i in range(len(encoded_d))]
    return final, final[0][0].size()[0]

def add_label(encoded_d, tag):
    labeled_d= [(ex,tag) for ex in encoded_d]
    return labeled_d

def struct_data(filename,tag, limit_train):
    structured_data = read_data(filename)
    structured_data = encode_data(structured_data)
    structured_data = add_label(structured_data, tag)

    return structured_data[0:limit_train] , structured_data[limit_train:]



class LSTMTagger(nn.Module):

    def __init__(self, input_size, out_size_lstm, hidden_size):
        super(LSTMTagger, self).__init__()
        self.out_size_lstm = out_size_lstm
        self.lstm = nn.LSTM(input_size, out_size_lstm)
        self.linear1 = nn.Linear(out_size_lstm, hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence.unsqueeze(0))
        lin1 = self.linear1(lstm_out)
        tanh = self.tanh(lin1)
        lin2 = self.linear2(tanh)
        soft = self.softmax(lin2)
        return soft

    def validate_model(self, dev_set, cuda, device, criterion):
        """
        This function validates the model.
        :param dev_set: data loader with dev set.
        :param cuda: boolean - true if gpu is aviable, false otherwise.
        :param device: cpu or gpu.
        :param criterion: loss function
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
                outputs = outputs.squeeze(0)
                loss = criterion(outputs, label)
                loss_list.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                for indx, (pr, lb) in enumerate(zip(predicted, label)):
                   correct += (pr == lb).sum().item()
                   total+=1

            if total != 0:
                acc = correct * 100 / total
            else:
                acc = 0
            return acc, np.sum(loss_list) / len(loss_list)


    def train_model(self, optimizer, criterion, data_train_loader, data_dev_loader, epochs, cuda, device):
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

        count = True
        val_acc_per_epoch = []
        val_loss_per_epoch = []

        local_loss = 0
        acc_list = []
        total_step = len(data_train_loader)

        for e in range(epochs):
            self.train()
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
                outputs = outputs.squeeze(0)
                loss = criterion(outputs, label)
                loss_list.append(loss.item())

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the accuracy

                _, predicted = torch.max(outputs.data, 1)
                for i, (p, l) in enumerate(zip(predicted, label)):
                    correct += (p == l).sum().item()
                    total+=1
            train_acc = 0
            if total != 0: train_acc = (correct / total) * 100

            train_loss = sum(loss_list) / len(loss_list)
            print('Epoch [{}/{}], Train_Loss: {:.4f}, Train_Accuracy: {:.2f}%'
                  .format(e + 1, epochs, train_loss, train_acc))

            dev_acc_loss = self.validate_model(data_dev_loader, cuda, device, criterion)
            print("Epoch [{}/{}] ,Dev_Loss:{:.4f},  Dev_Accuracy:{:.2f}%".format(e + 1, epochs, dev_acc_loss[1], dev_acc_loss[0]))
            val_acc_per_epoch.append(dev_acc_loss[0] / 100)
            val_loss_per_epoch.append(dev_acc_loss[1])
        return val_acc_per_epoch, val_loss_per_epoch


if __name__ == "__main__":
    """
    argv[1] path to positive examples file
    argv[2] path to negative examples file
    argv[3] limit (it should be MAX_RANDOM from gen_examples.)
    """
    #-----HYPERPARAMETERS-------
    batch_size = 100
    out_size_lstm = 150
    hidden_size = 50
    l_r = 0.001
    criterion = nn.CrossEntropyLoss()
    epochs = 10
    #----------------------


    limit_train = int(argv[3])
    # train_pos = struct_data(argv[1], TAG_INDX_POS)
    # train_neg = struct_data(argv[2], TAG_INDX_NEG)



    train_pos , dev_pos = struct_data(argv[1], TAG_INDX_POS, limit_train)
    train_neg, dev_neg = struct_data(argv[2], TAG_INDX_NEG, limit_train)
    train, size = resized_encoded_data(train_pos + train_neg)
    dev, seze_dev= resized_encoded_data(dev_pos + dev_neg)


    data_train_loader = dl.DataLoader(train, batch_size=batch_size, shuffle=True)
    data_dev_loader = dl.DataLoader(dev, batch_size=batch_size, shuffle=True)
    length = len(train[0][0])
    cud = torch.cuda.is_available()
    device = torch.device("cuda" if cud else "cpu")
    model =LSTMTagger(length, out_size_lstm, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r)

    if cud: model = model.cuda()
    val_acc_per_epoch, val_loss_per_epoch = model.train_model(optimizer,criterion,data_train_loader, data_dev_loader, epochs, cud, device)








