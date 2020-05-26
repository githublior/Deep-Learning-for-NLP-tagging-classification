# Students: Lior Shimon, id: 341348498
#           Lev Levin, id: 342480456

from sys import argv
from windowBasedTagger import tagger
from data import *
import torch.utils.data.dataloader as dl
import matplotlib.pyplot as plt
# ------------------------Constants-------------------------
PRE_TRAINED_OFF = "off"
PRE_TRAINED_ON = "on"
SUB_UNITS_OFF = "off"
SUB_UNITS_ON="on"

NER_ROUTINE ="ner"
POS_ROUTINE = "pos"

NER_TEST_DEST = "test.ner"
POS_TEST_DEST = "test.pos"

O_TAG = 'O'
# ----------------------------------------------------------


def plot_function(y, x,name_x,name_y,tittle,filename):
    """
    This function creates a plot graph of y as a function of x.(y is a list of values and x is a list
    of values).
    :param y: list with y values
    :param x: list with x values
    :param name_x: name of x axe.
    :param name_y:  name of y axe.
    :param tittle: graph tittle
    :param filename: path to file where to store the plot.
    :return:
    """
    #plt.plot(range(1, epoch_count + 1), train_accuracy_per_epoch, label="train")
    plt.plot(x, y, label=tittle)
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.title(tittle)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def perform_tagging_routine(routine_name, data_container):
    """
    This function initializes window based tagger with different parameters
    according to routine_name(pos or ner) trains the model with these parameters,
    plots two graphs of validation loss and accuracy, predicts with the model
    on examples from test set and stores the results to file.

    :param routine_name: routine name(POS_ROUTINE or NER_ROUTINE)
    :param data_container: an object that contains all needed data to train,validate, and test model.
    :return: None.
    """
    if routine_name == POS_ROUTINE:
        # ---------------hyper parameters-------------------
        epochs = 40
        hidden_size = 60
        l_r = 0.01
        torch.manual_seed(1)
        batch_size = 1000
        # --------------------------------------------------

        # -------------Other configurations for this routine------
        o_tag_indx = None
        output_test_path = POS_TEST_DEST
        # -----------------------------------------------------

    elif routine_name == NER_ROUTINE:
        # ---------------hyper parameters-------------------
        epochs = 1
        hidden_size = 10
        l_r = 0.01
        torch.manual_seed(1)
        batch_size = 1000
        # --------------------------------------------------

        # -------------Other configurations for this routine------
        o_tag_indx = data_container.labels_to_index[O_TAG]
        output_test_path = NER_TEST_DEST
        #-----------------------------------------------------
    input_size = data_container.Embedding_matrix.embedding_dim*5

    data_train_loader = dl.DataLoader(data_container.dataset_train,
                                      batch_size=batch_size, shuffle=True)
    data_dev_loader = dl.DataLoader(data_container.dataset_dev,
                                    batch_size=batch_size, shuffle=True)
    test_set = data_container.test_x

    criterion = nn.CrossEntropyLoss()
    cud = torch.cuda.is_available()
    device = torch.device("cuda" if cud else "cpu")

    model = tagger(data_container.Embedding_matrix,
                   input_size, hidden_size, len(data_container.labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_r,weight_decay=1e-5)
    if cud: model = model.cuda()

    val_acc_per_epoch, val_loss_per_epoch = model.train_model(optimizer, criterion, data_train_loader,
                                                              data_dev_loader, epochs, cud, device,
                                                              O_to_index=o_tag_indx,
                                                              miss_index=data_container.labels_to_index[MISS_TAG])

    plot_function(y = val_acc_per_epoch,x = range(len(val_acc_per_epoch)),name_x="epochs",name_y="accuracy"
                  ,tittle=routine_name + " accuracy",filename="plot_acc_" +routine_name)
    plot_function(y = val_loss_per_epoch,x = range(len(val_loss_per_epoch)),name_x="epochs",name_y="loss"
                  ,tittle=routine_name + " loss", filename="plot_loss_" +routine_name )

    model.test_model(test_set, output_test_path, data_container, cud, device)


if __name__ == "__main__":
    routine_name = argv[1]
    use_pre_emdeb = argv[2]
    use_sub_units = argv[3]
    path_train = argv[4]
    path_dev = argv[5]
    path_test = argv[6]
    if routine_name == POS_ROUTINE:
        delimiter = " "
    elif routine_name == NER_ROUTINE:
        delimiter = "\t"
    if use_pre_emdeb ==PRE_TRAINED_OFF and use_sub_units == SUB_UNITS_OFF:
        NER_TEST_DEST = "test1.ner"
        POS_TEST_DEST = "test1.pos"
        data_container = DataContainer(path_train,path_dev,path_test,use_sub_units=False,delim=delimiter)

    elif use_pre_emdeb == PRE_TRAINED_OFF and use_sub_units == SUB_UNITS_ON:
        NER_TEST_DEST = "test4_without_pre_trained.ner"
        POS_TEST_DEST = "test4_without_pre_trained.pos"
        data_container = DataContainer(path_train, path_dev, path_test, use_sub_units=True,delim=delimiter)

    elif use_pre_emdeb == PRE_TRAINED_ON and use_sub_units == SUB_UNITS_OFF:
        NER_TEST_DEST = "test3.ner"
        POS_TEST_DEST = "test3.pos"
        data_container = DataContainer(path_train,path_dev,path_test,
                                       pre_embd_words_path = argv[7],
                                       pre_embd_vectors_path = argv[8], use_sub_units=False,delim=delimiter)

    elif use_pre_emdeb == PRE_TRAINED_ON and use_sub_units == SUB_UNITS_ON:
        NER_TEST_DEST = "test4_with_pre_trained.ner"
        POS_TEST_DEST = "test4_with_pre_trained.pos"
        data_container = DataContainer(path_train, path_dev, path_test,
                                       pre_embd_words_path=argv[7],
                                       pre_embd_vectors_path=argv[8], use_sub_units=True,delim=delimiter)
    perform_tagging_routine(routine_name, data_container)
