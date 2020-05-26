# Lev Levin 342480456
# Lior Shimon 341348498

from sys import argv

from utils import *  # All relevant constant and helpful functions are located in this file.
import data_ac as d_ac
import data_b as d_b
import data_d as d_d
import tagger
import torch
import torch.utils.data.dataloader as dl
import torch.nn as nn
import pickle
def perform_routine(routine_name, data_container,repr,graph_file,model_file,container_file):
    """
    This function initializes tagger with different parameters
    according to routine_name(pos or ner) and repr - representation of sentences(a,b,c,d)
    and trains the model with these parameters, saves the model's parameters to file and saves
    data_container to file.

    :param routine_name: routine name(POS_ROUTINE or NER_ROUTINE)
    :param data_container: an object that contains all needed data to train,validate, and test model.
    :return: None.
    """


    if routine_name == POS_ROUTINE:
        # ---------------hyper parameters-------------------
        epochs = 5
        blstm_output = BLSTM_OUTPUT # will be doubled
        num_of_features_for_blstm = NUM_OF_FEATURES_FOR_BLSTM
        l_r = 0.01
        torch.manual_seed(1)
        batch_size = 1
        criterion = nn.CrossEntropyLoss()
        # l_r
        # Optimizer
        # --------------------------------------------------

        # -------------Other configurations for this routine------
        o_tag_indx = None
        # -----------------------------------------------------

    elif routine_name == NER_ROUTINE:
        # ---------------hyper parameters-------------------
        epochs = 5
        blstm_output = BLSTM_OUTPUT # will be doubled
        num_of_features_for_blstm = NUM_OF_FEATURES_FOR_BLSTM
        l_r = 0.0001
        torch.manual_seed(1)
        batch_size = 1
        criterion = nn.CrossEntropyLoss()
        #l_r
        #Optimizer
        # --------------------------------------------------

        # -------------Other configurations for this routine------
        o_tag_indx = data_container.labels_to_index[O_TAG]
        #-----------------------------------------------------

    data_train_loader = dl.DataLoader(data_container.dataset_train,
                                      batch_size=batch_size, shuffle=True)
    data_dev_loader = dl.DataLoader(data_container.dataset_dev,
                                    batch_size=batch_size, shuffle=True)


    cud = torch.cuda.is_available()
    device = torch.device("cuda" if cud else "cpu")
    # Creating the tagger.
    model = tagger.network(data_container.Embedding_matrix,
                           num_of_features_for_blstm, blstm_output, len(data_container.labels), repr,
                           max_word_len=d.get_maximum_word_length(),
                           if_cud=cud, device=device).to(device)
    #Optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r, weight_decay=L2_REGUL)
    if cud: model = model.cuda()

    val_acc_list, val_loss_list = model.train_model(optimizer, criterion, data_train_loader,
                                                              data_dev_loader, epochs, cud, device,
                                                              O_to_index=o_tag_indx,

                                                              miss_index=data_container.labels_to_index[MISS_TAG])

    # storing the data that may be used later for creating plot of a validation accuracy with pickle.
    data_to_store = (val_acc_list,repr,routine_name)
    # The name of the file is dynamic(the constants may be changed in utils and repr, and routine are parameters
    # for this function).
    with open(PREFIX_FOR_GRAPH_DATA + SEPERATOR + repr + SEPERATOR + routine_name,'wb+') as f:
        pickle.dump(data_to_store, f)
    #this list will be passed to the graph function(it graph path was passed)
    all_data_for_graph = [data_to_store]
    # if graph pass is not empty.
    if graph_file is not None:
        plot_function(graph_file,all_data_for_graph,NUM_OF_SENTENCES_TO_PRINT,"number of sentences / 100","accuracy")
    #saving pytorch model parameters to file.
    torch.save(model.state_dict(),model_file)
    # we are moving all sets to cpu before saving data_container so that if afterwards someone reloads
    # the file on machine without GPU
    # he would use it without getting errors.
    data_container.move_sets_to_cpu()
    with open(container_file, 'wb+') as f:
        pickle.dump(data_container,f)




if __name__ == "__main__":
    representation = argv[1]
    train_file = argv[2]
    model_file = argv[3]
    dev_file = argv[4]
    graph_file = argv[5]
    routine_type = argv[6]
    container_file = argv[7]
    # We ensure that graph file for pos will start with prefix 'pos' and graph file for ner
    # will start with 'ner' in order to not make mistakes such as running a model on pos set
    # but drawing on ner graph.


    # if not graph_file.endswith(routine_type):
    #       raise ValueError("routing type doesn't match graph file!")
    if graph_file == GRAPH_FILE_OFF: graph_file = None
    if routine_type == NER_ROUTINE:
        delimiter = DELIM_NER
    elif routine_type == POS_ROUTINE:
        delimiter = DELIM_POS
    else:
        raise NotImplementedError("NOT IMPLEMENTED YET")
    if representation == REPR_A:
        d = d_ac.DataContainer_ac(train_file,dev_file,delim=delimiter)
    elif representation == REPR_C:
        d =d_ac.DataContainer_ac(train_file,dev_file,delim=delimiter,use_sub_units=True)
    elif representation == REPR_B:
        d = d_b.DataContainer_b(train_file,dev_file,delim = delimiter)
    elif representation == REPR_D:
        d  = d_d.DataContainer_d(train_file,dev_file,delim = delimiter)

    else:
        raise NotImplementedError("NOT IMPLEMENTED YET")
    perform_routine(routine_type,d,representation,graph_file,model_file,container_file)
