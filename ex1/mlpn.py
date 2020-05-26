import numpy as np
import loglinear as ll
STUDENT = {'name': 'Lior Shimon_Lev Levin',
    'ID': '341348498_342480456'}


def classifier_output(x, params):
     # YOUR CODE HERE

    #creating convinient lists for W0, W1,..,Wn parameters and b1,b2,...,bn parameters
    param_W = [params[i] for i in range(0, len(params), 2)]
    param_B = [params[i] for i in range(1, len(params), 2)]

    hi =  x
     #performing forward propagation
    for i in range(len(param_W) - 1):
        hi = np.tanh(np.dot(hi, param_W[i]) + param_B[i])
    #computing the output in the last layer(with softmax).
    probs = ll.softmax(np.dot(hi,param_W[len(param_W)-1]) + param_B[len(param_B) - 1 ])
    return probs




def predict(x, params):
    return np.argmax(classifier_output(x, params))




def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    y_hat = classifier_output(x, params)

    #y_true is a vector label for this example.
    y_true = np.zeros_like(y_hat)
    y_true[y] = 1

    y_result = y_hat - y_true

    param_W = [params[i] for i in range(0,len(params),2)]
    param_B = [params[i] for i in range(1,len(params),2)]

    #Z is a list with x,z1,z2,...,zn values where zi = W_i*tanh(z_i-1) +b_i
    Z  = []
    Z.append(np.dot(x,param_W[0]) + param_B[0])
    for i in range(1,len(param_B)):
        Z.append(np.dot(np.tanh(Z[i - 1]),param_W[i])+ param_B[i])


    #we reverse list with parameters and z because we will compute gradients of b_i starting from
    #computing gb_n and then  gb_n-1 and then gb_n-2(because the formula for gb_i uses value of gb_i-1
    #and also param_W_i-1
    param_W.reverse()
    param_B.reverse()
    Z.reverse()
    gB = []
    gB.append(y_result)

    for i in range(1,len(param_B)):
        gB.append(np.dot(param_W[i - 1],gB[i - 1]) * (1 - np.square(np.tanh(Z[i]))))

    #reversing parameters lists back.
    param_W.reverse()
    param_B.reverse()
    Z.reverse()
    gB.reverse()

    #gradients for W1,W2,..,Wn.
    gW = []
    gW.append(np.outer(x, gB[0]))
    for i in range(1,len(param_B)):
        gW.append(np.outer(np.tanh(Z[i - 1]),gB[i]))

    loss = - np.log(y_hat[y])
    gradients = []
    for i in range(len(param_W)):
        gradients.append(gW[i])
        gradients.append(gB[i])

    return loss, gradients

def glorot_function(in_dim,in_out):
    """
        This function gives glorot border of uniform distribution.
        That is, it computes e_glorot for [-e_glorot,e_glorot] - range for uniform distribution
        :param in_dim: input dimension
        :param out_dim: output dimension.
        :return: e_glorot
        """
    return np.sqrt(6) /(np.sqrt(in_out + in_dim))

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.

    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    np.random.seed(1)
    params = []

    #we need to handle case where dims in size of 2 because the general loop doesn't handle this.
    if len(dims) == 2:
        #initializing parameters using uniform distribution and glorot.
        params.append(np.random.uniform(low=-glorot_function(dims[0],dims[1]), high=glorot_function(dims[0],dims[1]),size=(dims[0],dims[1])))
        params.append(np.random.uniform(low=-glorot_function(dims[0],dims[1]), high=glorot_function(dims[0],dims[1]),size=dims[1]))
    else:
        for i in range(len(dims) - 1):
            # initializing parameters using uniform distribution and glorot.
            params.append(np.random.uniform(low=-glorot_function(dims[i],dims[i+1]), high=glorot_function(dims[i],dims[i+1]),size=(dims[i],dims[i+1])))
            params.append(np.random.uniform(low=-glorot_function(dims[i],dims[i+1]), high=glorot_function(dims[i],dims[i+1]),size=dims[i+1]))


    return params


if __name__ == '__main__':
    from grad_check import gradient_check
    W, b, U, b_tag,W2,b2,W3,b3 = create_classifier([3, 21,71, 31, 50])


    def _loss_and_W_grad(W):
        """
            This function creates random x vector, canculates on it loss and gradients for w and
            returns loss and w-gradients that were calculated for random x vector.
            """
        global b
        global U
        global b_tag
        global W2
        global b2
        global W3
        global b3
        loss, grads = loss_and_gradients([5, 2, 3], 3, [W, b, U, b_tag,W2,b2,W3,b3])
        return loss, grads[0]


    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag
        global W2
        global b2
        global W3
        global b3
        loss, grads = loss_and_gradients([6, 2, 3], 7, [W, b, U, b_tag,W2,b2,W3,b3])
        return loss, grads[1]


    def _loss_and_U_grad(U):
        global W
        global b
        global b_tag
        global W2
        global b2
        global W3
        global b3
        loss, grads = loss_and_gradients([7, 0, 0], 2, [W, b, U, b_tag,W2,b2,W3,b3])
        return loss, grads[2]


    def _loss_and_b_tag_grad(b_tag):
        global W
        global b
        global U
        global W2
        global b2
        global W3
        global b3
        loss, grads = loss_and_gradients([1, 2, 100], 3, [W, b, U, b_tag,W2,b2,W3,b3])
        return loss, grads[3]
    def _loss_and_W2_grad(W2):
        global W
        global b
        global b_tag
        global U
        global b2
        global W3
        global b3
        loss, grads = loss_and_gradients([7, 0, 0], 2, [W, b, U, b_tag,W2,b2,W3,b3])
        return loss, grads[4]


    def _loss_and_b2_grad(b2):
        global W
        global b
        global U
        global b_tag
        global W2
        global W3
        global b3
        loss, grads = loss_and_gradients([1, 2, 100], 3, [W, b, U, b_tag,W2,b2,W3,b3])
        return loss, grads[5]


    def _loss_and_W3_grad(W3):
        global W
        global b
        global U
        global b_tag
        global W2
        global b2
        global b3
        loss, grads = loss_and_gradients([1, 2, 100], 3, [W, b, U, b_tag, W2, b2, W3, b3])
        return loss, grads[6]


    def _loss_and_b3_grad(b3):
        global W
        global b
        global U
        global b_tag
        global W2
        global W3
        global b2
        loss, grads = loss_and_gradients([1, 2, 100], 3, [W, b, U, b_tag, W2, b2, W3, b3])
        return loss, grads[7]


    def text_to_unigrams(text):
        return ["%s" % c1 for c1 in zip(text[1:])]

    for _ in range(100):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
        b_tag = np.random.randn(b_tag.shape[0])
        W2 = np.random.randn(W2.shape[0], W2.shape[1])
        b2 = np.random.randn(b2.shape[0])
        W3 = np.random.randn(W3.shape[0], W3.shape[1])
        b3 = np.random.randn(b3.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
        gradient_check(_loss_and_U_grad, U)
        gradient_check(_loss_and_b_tag_grad, b_tag)
        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_b2_grad, b2)
        gradient_check(_loss_and_W3_grad, W3)
        gradient_check(_loss_and_b3_grad, b3)

    # classier_params = create_classifier([5,10,15,20,30,10,5,3,32,11])
    # for _ in range(10):
    #     for i in range(0,len(classier_params) - 1,2):
    #         s1 = classier_params[i].shape[0]
    #         s2 = classier_params[i].shape[1]
    #         classier_params[i] = np.random.rand(classier_params[i].shape[0],classier_params[i].shape[1])
    #         classier_params[i+1] = np.random.rand(classier_params[i+1].shape[0])
    #
    #     for i in range(len(classier_params)):
    #         gradient_check(loss_and_grad_i,classier_params[i])




