
import loglinear as ll
import numpy as np
import random
import mlp1
from collections import Counter

STUDENT = {'name': 'Lior Shimon_Lev Levin',
    'ID': '341348498_342480456'}

def read_data(fname):
    """
      Reading the data set from file.
      :param fname: file name
      :return: nd array that represents data.
      """
    file = open(fname,encoding='utf8')
    data = []
    
    for line in file:
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return np.array(data)


def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def text_to_unigrams(text):
    return ["%s" % c1 for c1 in zip(text[1:])]

def feats_to_vec(features):  #features = train[j][1]-- the text of one example.
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    # we transform the features of each example - the Text - to a vector that represent
    # the number of occurence of each pairs of letter present in vocab.
    vec = np.zeros(vocab_size)
    
    for i in features:
        if i in vocab:
            vec[F2I[i]] += 1
    return vec


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        
        
        label = L2I[label]
        features = feats_to_vec(features)
        
        if mlp1.predict(features, params) == label:
            good +=1
        else: bad +=1
        
        pass
    accuracy = good / (good + bad)

    return accuracy


def train_classifier(train_data, dev_data, num_iterations , learning_rate , params):
    """
        Create and train a classifier, and return the parameters.
        
        train_data: a list of (label, feature) pairs.
        dev_data  : a list of (label, feature) pairs.
        num_iterations: the maximal number of training iterations.
        learning_rate: the learning rate to use.
        params: list of parameters (initial values)
        """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = np.copy(feats_to_vec(features))  # convert features to a vector.
            y = np.copy(L2I[label])  # convert the label to number if needed.
            loss, grads = mlp1.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            
            # new_grad = np.array(grads)
            W, b, U, b_tag = params
            W -= learning_rate * grads[0]
            b -= learning_rate * grads[1]
            U -= learning_rate * grads[2]
            b_tag -= learning_rate * grads[3]
            
            params = [W , b , U , b_tag]
        
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print('iteration :' + str(I) +
              'train_ loss: ' + str(train_loss) +
              'train_acc:'+ str(train_accuracy) +
              ' dev_acc:' , str(dev_accuracy))
    return params

# for part 4
def save_test_prediction(output_path, test_predictions):
    with open(output_path, "w") as output_file:
        output_file.write("\n".join(map(str, test_predictions)))



if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    train_data = read_data('train')
    dev_data = read_data("dev")
    test_data = read_data("test")
    
    TRAIN = [(l, text_to_bigrams(t)) for l, t in train_data]
    DEV = [(l, text_to_bigrams(t)) for l, t in dev_data]
    TEST = [(l, text_to_bigrams(t)) for l, t in test_data]
    #-----------------------------------------------------------------
    # UNIGRAM
    # TRAIN = [(l, text_to_unigrams(t)) for l, t in train_data]
    # DEV = [(l, text_to_unigrams(t)) for l, t in dev_data]
    #------------------------------------------------------------------
    fc = Counter()
    for l, feats in TRAIN:
        fc.update(feats)
    
    #  most common bigrams in the training set.
    #  -----------------------HYPERPARAMETER--------------------------------
    vocab_size = 600
    epochs = 14
    learning_rate = .001
    hid_layer = int(1.2 * vocab_size)
    vocab = set([x for x, c in fc.most_common(vocab_size)])
    
    # label strings to IDs
    L2I = {l: i for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
    I2L = {i: l for i, l in enumerate(list(sorted(set([l for l, t in TRAIN]))))}
    # feature strings (bigrams) to IDs
    F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
    # ...
    
    params = mlp1.create_classifier(vocab_size,hid_layer ,  len(L2I))
    trained_params = train_classifier(TRAIN, DEV, epochs, learning_rate, params)


# #__________PART 4 ____________
#after training the model , the parameters are supposely set to reach max accuracy.
    #
    # test_predictions = []
    # for label, features in TEST:
    #     x = feats_to_vec(features)
    #     test_predictions.append(I2L[mlp1.predict(x, params)])   # later ll change to nn.
    #
    # save_test_prediction("test.pred", test_predictions)
