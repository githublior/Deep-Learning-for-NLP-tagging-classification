Lev Levin, 342480456
Lior Shimon, 341348498

NEEDED_FILES:  tagger1.py,data.py,windowBasedTagger.py(in the same directory)

		
HOW_TO_RUN:
	run with command: " tagger1.py <tagger_option> <pre_embedding_on_off> <sub_units_on_off> <path_to_train> <path_to_dev> <path_to_test> <path_to_pre_embedding_words> <path_to_pre_vectors>
	where:
		<tagger_option>:  pos - if you want to train pos model.
		          		ner - if you want to train ner model.
		<pre_embedding_on_off>:  on - if you want to use pre embedding. In this case it's necessary to provide <path_to_pre_embedding_words> and <path_to_pre_vectors>
				            off - if you want to initialize embedding matrix randomly.
		<sub_units_on_off>: on - if you want model to use suffixes and preffixes of words.
				 off - if you want use only words.
		<path_to_train> - path to file with train data.
		<path_to_dev> - path to file with validation data.
		<path_to_test> - path to file with test data.
		<path_to_pre_embedding_words> - path to file which contains pre embedding words. It's neccessary to provide this option if you choosed 'on' in <pre_embedding_on_ff>.
					           If you chose 'off', than the program will ignore this argument.
		<path_to_pre_vectors> - path to file which containts pre embedding vectors. It's neccessary to provide this option if you choosed 'on' in <pre_embedding_on_ff>.
					           If you chose 'off', than the program will ignore this argument.

	NOTE: Parameters should fit to the options. For example, if you choose 'pos' option, <path_to_train> need to be a path to pos train data.
	Example of running: "ner on off train dev test embedding/words embedding/vectors"
	
	
OUTPUTS:
	The program produces: 1) file with test predictions. The name of the file depends on the option you choose. For example, for command: "ner off off train dev test", it will produce test1.ner.
						      That is, a file name will fit to the part of the assigmnent. The program will detect to which part provided arguments are related.
	
			        2) plot_acc_<tagger_option>.png - graph showing an accuracy as a function of a number of iterations. <tagger_option> is a choosed option: 'ner' or 'pos'.
			        3) plot_loss_<tagger_option>.png - graph showing a loss as a function of a number of iterations. <tagger_option> is a choosed option: 'ner' or 'pos'.


HOW TO RUN FOR SPECIFIC PART IN THE ASSIGMENT:
	each part of the assigment requires specific options. For instance, part1 requires <pre_embedding_on_off> <sub_units_on_off> both to be 'off'. So, for part1, run command with these
	options set to 'off'. For part4, for example,  <sub_units_on_off> is always 'on' and other options could be switched. Therefore just choose the options that fit to the assigment part you want to run.


