# Lev Levin 342480456
# Lior Shimon 341348498

# All relevant constant and helpful functions are located in this file.
from utils import *
import data_ac as d_ac
import data_b as d_b
import data_d as d_d
import tagger
import torch
import torch.utils.data.dataloader as dl
import torch.nn as nn

def perform_routine(routine_name, data_container,repr,model_file):
    """
    This function loads the model from file and predicts on test set which is stored
    in data_container and saves the predictions to file 'PREFIX_FOR_OUTPUT_TEST + SEPERATOR + repr +
     SEPERATOR + routine_name'
    :param routine_name:
    :param data_container:
    :param repr:
    :param model_file:
    :return:
    """
    if routine_name == POS_ROUTINE:
        # ---------------hyper parameters-------------------
        hidden_size = BLSTM_OUTPUT #PAY ATTENTION! HERE YOU NEED TO PUT WHAT YOU USED IN THE TRAINING!
        torch.manual_seed(1)
        # --------------------------------------------------
    elif routine_name == NER_ROUTINE:
        # ---------------hyper parameters-------------------
        hidden_size = BLSTM_OUTPUT #PAY ATTENTION! HERE YOU NEED TO PUT WHAT YOU USED IN THE TRAINING!
        torch.manual_seed(5)
        # --------------------------------------------------

    test_set = data_container.test_x

    cud = torch.cuda.is_available()
    device = torch.device("cuda" if cud else "cpu")

    model = tagger.network(data_container.Embedding_matrix,
                           EMBEDING_LEN, hidden_size, len(data_container.labels), repr,
                           max_word_len=d.get_maximum_word_length(),
                           if_cud=cud, device=device)

    model.load_state_dict(torch.load(model_file))
    if cud: model = model.cuda()
    model = model.to(device)
    output_f_name = PREFIX_FOR_OUTPUT_TEST + SEPERATOR + repr + SEPERATOR + routine_name
    model.test(test_set,output_f_name,data_container,cud,device,routine_name)




if __name__ == "__main__":
    """
    Arguments for main:
        argv[1]: representation
        argv[2]: model_file 
        argv[3]: input_file
        argv[4]: routine_type
        argv[5]: container_file
    """
    from sys import argv
    representation = argv[1]
    model_file = argv[2]
    input_file = argv[3]
    routine_type = argv[4]
    container_file = argv[5]


    if routine_type == NER_ROUTINE:
        delimiter = DELIM_NER
    elif routine_type == POS_ROUTINE:
        delimiter = DELIM_POS
    else:
        raise NotImplementedError("NOT IMPLEMENTED YET")
    with open(container_file, 'rb') as f:
        d = pickle.load(f)
    #Here we put the test set of the passed file to data container.
    d.replace_test_set(input_file)
    perform_routine(routine_type,d,representation,model_file)