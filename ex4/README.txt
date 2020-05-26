Lev Levin, 342480456
Lior Shimon, 341348498
GENERAL DESCRIPTION:  
	the program trains a network described in the paper -A Decomposable Attention Model for Natural Language Inference- from Parikh et al. The full paper is available at: https://arxiv.org/pdf/1606.01933v1.pdf
NEEDED_FILES: 
	main.py network.py dataloader.py  utils.py 
NEEDED_DIRECTORIES:  
	-cached_data  - containing SNLI Training set, Dev set and Test set
	-cached_vector - containing GLove300d-42B embedding

	All thoses files and directories need to be in the same directory.


HOW_TO_RUN:  on python 3.7, run with command:  main.py 

OUTPUTS:
		files: 
		- serialized object "dev_acc_file" containing a list with dev accuracies(dev accuracy per epoch)
		- serialized object "tr_acc_file" containing a list with train accuracies(train accuracy per epoch)
		- log_ass4.log file - logging file of running program.

		console and log output:
		- the program prints train, dev losses and train,dev accuracies on all train/dev data set each training epoch.
		- after training is finished, the programs prints accuracy on test set.

