Lev Levin, 342480456
Lior Shimon, 341348498

NEEDED_FILES:  experiment.py

		
HOW_TO_RUN:
	run with command: " experiment.py  <path_to_positive_example> <path_to_negative_example> <delimiter_number>
	where:
		<path_to_positive_train> - path to file with positive example train data and dev data.
		<path_to_negative_train>  -path to file with negative example train data and data.
		<delimiter_number> - all examples under this number goes in the training set, and all the examples over this number goes to validation set 


		
	NOTE: Parameters should fit to the options. 
	
	
OUTPUTS:
	for each bach size of each epoch, the program print out the loss and the accuracy of the model on the training set (composed of positive and negative examples)
	and on the validation set  (composed of positive and negative examples).

