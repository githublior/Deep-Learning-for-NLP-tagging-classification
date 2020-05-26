import numpy as np

STUDENT = {'name': 'Lior Shimon_Lev Levin',
    'ID': '341348498_342480456'}

#-------FINISHED---------
def softmax(x):
    """
        Compute the softmax vector.
        x: a n-dim vector (numpy array)
        returns: an n-dim vector (numpy array) of softmax values
        """
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.

    # Softmax using normalization of the data

    max = np.max(x)
    expo = np.exp(x - max)
    expo_sum = np.sum(np.exp(x - max))
    return expo / expo_sum

#------FINISHED-------
def classifier_output(x, params):
    """
        Return the output layer (class probabilities)
        of a log-linear classifier with given params on input x.
        """
    # YOUR CODE HERE.
    W, b = params
    probs = softmax(np.dot(x, W) + b)

    return probs


def predict(x, params):
    """
        Returnss the prediction (highest scoring class id) of a
        a log-linear classifier with given parameters on input x.

        params: a list of the form [(W, b)]
        W: matrix
        b: vector
        """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
        Compute the loss and the gradients at point x with given parameters.
        y is a scalar indicating the correct label.

        returns:
        loss,[gW,gb]

        loss: scalar
        gW: matrix, gradients of W
        gb: vector, gradients of b
        """



    W, b = params
    # YOU CODE HERE


    y_hat = classifier_output(x, params)
    #y indicates a correct label.
    loss = - np.log(y_hat[y])
    y_true = np.zeros_like(y_hat)
    y_true[y] = 1
    y_result = y_hat - y_true
    gW = np.outer(x,y_result)
    gb =  y_result
    return loss, [gW, gb]


def glorot_function(in_dim, out_dim):
    """
    This function gives glorot border of uniform distribution.
    That is, it computes e_glorot for [-e_glorot,e_glorot] - range for uniform distribution
    :param in_dim: input dimension
    :param out_dim: output dimension.
    :return: e_glorot
    """
    return np.sqrt(6) / (np.sqrt(out_dim + in_dim))
def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print (test1)
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print (test2)
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002]))
    print (test3)
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(5,8)

    def _loss_and_W_grad(W):
        """
        This function creates random x vector, canculates on it loss and gradients for w and
        returns loss and w-gradients that were calculated for random x vector.
        """
        global b
        loss,grads = loss_and_gradients([1,2,3],[W,b])
        return loss,grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss,grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)
