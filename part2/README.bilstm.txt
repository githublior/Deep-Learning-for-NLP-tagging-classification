Lev Levin, 342480456
Lior Shimon, 341348498

NEEDED_FILES:  	bilstmTrain.py, bilstmPredict.py, data_ac.py, data_b.py, data_d.py, utils.py, tagger.py(in the same directory)

		
HOW_TO_RUN:
	1.) bilstmTrain.py:
		run with command:  bilstmTrain.py <representation_type> <path_to_train> <model_file> <path_to_dev> <graph_option> <tagging_option> <path_to_data_container>

	where:
			<representation_type> - a, b, c or d. choosing the type of representation of the sentences.
			<path_to_train> - path to file with train data.
			<model_file> - path to file to save the model
			<path_to_dev> - path to file with validation data.
			<graph_option> - Write a name of a graph file in this option parameter if you want to create .png file that shows a graph with 
					accuracy on validation set as a function of number of sentences / 100 outputed by the trained model.
					If you do not want to create the graph file, write 'off' in this option.
			<tagging_option> -  pos - if you want to train pos model.  
					   ner - if you want to train ner model.
			<path_to_data_container> - path to file to save serialized object that contains model-related data(vocab,word_to_index,label_to_index etc.) 
						It will be used by bilstmPredict.py to reload this data.

		
	2.) bilstmPredict.py:
		run with command:  bilstmPredict.py <representation_type> <model_file> <input_file> <tagging_option> <data_container_file>

		where:
			<representation_type> - a, b, c or d. choosing the type of representation of the sentencees.
			<model_file> - the file to load the model
			<input_file> - the test file
			<tagging_option> -  pos - if you want to train pos model.  ner - if you want to train ner model.
			<data_container_file> - the file to load serialized object with model-related data(vocab,word_to_index,label_to_index etc.).



	NOTE:
		Parameters should fit to the options. For example, if you provide 'pos' in <tagging_option> option, than <input_file> mush have test set for pos tagging.
		Examples of running: bilstmTrain.py a train my_model dev my_output_graph pos data_container_a_pos
				  bilstmpredict.py a my_model test_file pos container_a_pos
	          	<data_container_file> and <model_file> must be the results from the same run of bilstmTrain.py. Using files from different runs may lead to undefined behaviour.
	          	Note, because we save state_dict of the pytorch model and not model itself, bilstmPredict.py and bilstm.Train.py initialize hyper parameters (hidden sizes or
		embedding  len) independently. Therefore if you want to run bilstmPredict.py, make sure that it has the same hyper parameters that were used in training(all common parameters can be assigned in 
		utils.py file and both programs bilstmPredict.py and bilstmTrain.py initialize model with them)
	
	
OUTPUTS:
	1.) bilstmTrain.py:
		prints to console: 
			       the program prints to console train and dev losses and accuracies while training each 500 sentences(for train set it computes loss, and acc cumulatively,that is,
			       each new 500 sentences it computes loss and acc for thess new sentences together with all previously seen sentences and for dev set, each 500 sentences it computes
			       loss and acc for the whole dev set)
		created files: 
		 	       1. <model_file> -  file that stores pytorch model.
			       2. <data_container_file> - file that stores object with data-related information.
 			       3. if <graph_option> is not 'off', and the name was passed, than the graph 'png' file with function of training accuracy as num of sentences / 100  is created.
			       4. File with name 'graphdata_<representation_type>_<tagging_option>' - which contains serialized object of tupple (validation_accuracy_list, representation_type, tagging_option).
			           It can be used to draw graph that describes this data.
			            Example for such a file: graphdata_c_ner

	2.) bilstmPredict.py:
		output to console: 
			       the program has no output to console. 
		created files:
		 	       1.  file with predictions for <input_file> with name 'test4_<representation_type>_ <tagging_option>'	
			           Examples for such a file: test4_a_pos, test4_b_ner
		


			
			
	
