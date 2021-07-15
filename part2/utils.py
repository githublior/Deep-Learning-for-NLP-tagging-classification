import matplotlib.pyplot as plt
import pickle
import os
#------Constants-----------
REPR_A = 'a'
REPR_B = 'b'
REPR_C = 'c'
REPR_D = 'd'
DEV_POS_FILE = 'pos/dev'
PRE_TRAINED_OFF = "off"
PRE_TRAINED_ON = "on"
SUB_UNITS_OFF = "off"
SUB_UNITS_ON="on"

NER_ROUTINE ="ner"
POS_ROUTINE = "pos"

NER_TEST_DEST = "test.ner"
POS_TEST_DEST = "test.pos"
MISS_TAG = "111"


NUM_OF_SENTENCES_TO_PRINT = 500
O_TAG = 'O'
DELIM_POS = " "
DELIM_NER = "\t"
GRAPH_FILE_OFF = 'off'
REPR_TO_COLOR = { REPR_A:'yellow', REPR_B:'green', REPR_C:'blue', REPR_D:'red'}
PREFIX_FOR_GRAPH_DATA = "graphdata"
PREFIX_FOR_DATA_CONT ="data_container"
SEPERATOR = "_"

PREFIX_FOR_OUTPUT_TEST = "test4"

#-------------------------model's parameters
BLSTM_OUTPUT = 100
EMBEDING_LEN = 50
#for a and c it has to be EMBEDING_LEN!!!
NUM_OF_FEATURES_FOR_BLSTM = 50
L2_REGUL  = 0
DROPOUT=0.3
#--------------------------

def plot_function(graph_file, all_graph_data,x_frequency, x_laybel_name, y_laybel_name):
    """
    plots all functions reccorded at all_graph_data.
    all_graph_data it's a list with tupples (val_acc_list,repr,routine_name)
    :param graph_file:
    :param all_graph_data:
    :param x_frequency:
    :param x_laybel_name:
    :param y_laybel_name:
    :return:
    """
    #plt.plot(range(1, epoch_count + 1), train_accuracy_per_epoch, label="train")

    # for example, if x_frequency is 500, than this list will be [500,1000,1500,2000,...,]
    x_list = list(range(x_frequency,len(all_graph_data[0][0]) * x_frequency + 1,x_frequency))
    x_list = [x/100 for x in x_list]
    for y_list, label_name, routine_name in all_graph_data:
        plt.plot(x_list, y_list, REPR_TO_COLOR[label_name], label=label_name)
        plt.title(routine_name)
        plt.legend(loc="upper left")
        plt.xlabel(x_laybel_name)
        plt.ylabel(y_laybel_name)

    plt.savefig(graph_file)
    plt.show()




