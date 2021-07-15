import xeger
import random
from sys import argv

# -----------------------Constants---------------------------
MIN_RANDOM = 90  # minimum allowed length for each example.
MAX_RANDOM = 100  # maximum allowed length for each example.
NUM_OF_EXAMPLES = int(argv[3])
POS_LANGUAGE_REGEX = r"[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+"
NEG_LANGUAGE_REGEX = r"[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+"
# -----------------------------------------------------------

if __name__ == "__main__":
    """
        :arg: argv[1] - path to file for positive examples(will be created)
        :arg: argv[2] - path to file for negative examples(will be created)
        """
    pos_path = argv[1]
    neg_path = argv[2]
    pos_file = open(pos_path, "w+")
    neg_file = open(neg_path, "w+")
    # Loop that creates and writes negative examples to neg_file
    # and positive examples to pos_file.
    for i in range(NUM_OF_EXAMPLES):
        x_pos = random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('a',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('b',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('c',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('d',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))
        x_neg = random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('a',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('c',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('b',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices('d',k= random.randint(MIN_RANDOM,MAX_RANDOM))+random.choices(range(1,10),k= random.randint(MIN_RANDOM,MAX_RANDOM))
        x_pos_str = ""
        for x in x_pos: x_pos_str += str(x)
        x_neg_str = ""
        for x in x_neg: x_neg_str += str(x)


        pos_file.write(x_pos_str  + "\n")
        neg_file.write(x_neg_str + "\n")

        # x_pos = xeger.Xeger(limit=random.randint(MIN_RANDOM, MAX_RANDOM))
        # pos_file.write(x_pos.xeger(POS_LANGUAGE_REGEX) + "\n")
        #
        # x_neg = xeger.Xeger(limit=random.randint(MIN_RANDOM, MAX_RANDOM))
        # neg_file.write(x_neg.xeger(NEG_LANGUAGE_REGEX) + "\n")

    pos_file.close()
    neg_file.close()








