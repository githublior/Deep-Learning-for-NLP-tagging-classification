#Lev Levin 342480456
#Lior Shimon 341348498
import dataloader
import torchtext
import math
import torch
import torch.nn as nn
import network
import logging
from utils import *
import pickle
import datetime

def routine():
    """
    The routing for training the decomposable attention model.
    :param saved_weights_file: - file with pytorch state_dict to load already trained weights and
            continue training on them.
    :return:None.
    """
    logging.basicConfig(level=logging.INFO, filename=LOGGER_NAME + ".log")

    text = torchtext.data.Field(lower=True, batch_first=True)
    label = torchtext.data.Field(lower=True, batch_first=True)
    
    #100 random vectors, one of them will be assigned randomly to oov words.
    oov = None
    def handle_oov(t):
        """
        Function that handles the case when a word doesn't appear in glove vocab.
        one of 100 oov vectors will be randomly assigned as a vector of the words.
        :param t: not used
        :return: vector for word that doesn't appear in a vocab.
        """
        n = torch.randint(low=1, high=101, size=(1,))
        t_ret = oov[-n.item()]
        #return t_ret
        return t_ret
    # loading glove. If the cache folders or vectors are not in the project folder than the cache folder and vectors
    # will be download from internet
    glove = torchtext.vocab.GloVe(name= '42B', dim=emb_dim, unk_init=handle_oov, cache='cached_vectors')

    oov = torch.empty(100, emb_dim).normal_(0,1)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    batched_data = dataloader.SNLI.iters( batch_size=16, device=device, vectors=glove,root='cached_data')
    embed = nn.Embedding.from_pretrained(batched_data[0].dataset.fields['premise'].vocab.vectors.clone(),freeze=True)
    embed = embed.to(device)
    # print(glove.get_vecs_by_tokens(["the"], lower_case_backup=True))r

    # hyperparameters
    learn_rate = 0.05
    epochs = 300
    out_proj_size = 200
    out_lin_size = 200
    label_size = 3
    criterion = nn.CrossEntropyLoss()
    cuda = torch.cuda.is_available()

    model = network.Network(embed, out_proj_size, out_lin_size, label_size)
    model = model.to(device)

    optimizer = torch.optim.Adagrad(model.parameters(), lr=learn_rate,weight_decay=1e-5)
    model.train_model(optimizer, criterion,batched_data[0], batched_data[1], epochs, cuda, device, model_file)
    logging.info(100*"-" + "\n")
    logging.info(f"Training was finished at {datetime.datetime.now()}")
    logging.info("Predicting on test...")
    model.test(batched_data[2],cuda,device,criterion)



if __name__ == "__main__":
        routine()


