import numpy as np
import loglinear as ll

STUDENT = {'name': 'Lior Shimon_Lev Levin',
    'ID': '341348498_342480456'}


def classifier_output(x, params):
    # params =  W, b, U, b_tag.
    # YOUR CODE HERE.
    W, b, U, b_tag = params
    probs = ll.softmax(np.dot(np.tanh(np.dot(x, W) + b), U) + b_tag)
    return probs


def predict(x, params):
    """
        params: a list of the form [W, b, U, b_tag]
        """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
        params: a list of the form [W, b, U, b_tag]
        
        returns:
        loss,[gW, gb, gU, gb_tag]
        
        loss: scalar
        gW: matrix, gradients of W
        gb: vector, gradients of b
        gU: matrix, gradients of U
        gb_tag: vector, gradients of b_tag
        """
    # YOU CODE HERE
    W, b, U, b_tag = params
    y_hat = classifier_output(x, params)

    #one-hot encoded label.
    y_true = np.zeros_like(y_hat)
    y_true[y] = 1

    V = np.dot(x, W) + b
    y_result = y_hat - y_true
    #gradients for U matrix.
    gU = np.outer(np.tanh(V), y_result)

    loss = - np.log(y_hat[y])
    # gradients for W matrix.
    gW = np.outer(x, np.dot(U, (y_hat - y_true)) * (1 - np.square(np.tanh(V))))
    #gradients for b vector.
    gb = np.dot(U, (y_hat - y_true)) * (1 - np.square(np.tanh(V)))
    # gradients for b_tag vector.
    gb_tag = y_result
    return loss, [gW, gb, gU, gb_tag]

def glorot_function(in_dim,in_out):
    """
        This function gives glorot border of uniform distribution.
        That is, it computes e_glorot for [-e_glorot,e_glorot] - range for uniform distribution
        :param in_dim: input dimension
        :param out_dim: output dimension.
        :return: e_glorot
        """
    return np.sqrt(6) /(np.sqrt(in_out + in_dim))

def create_classifier(in_dim, hid_dim, out_dim):
    """
        returns the parameters for a multi-layer perceptron,
        with input dimension in_dim, hidden dimension hid_dim,
        and output dimension out_dim.
        
        return:
        a flat list of 4 elements, W, b, U, b_tag.
        """

    np.random.seed(1)
    #initialization of the parameters with random uniform distribution and glorot border.
    W = np.random.uniform(low=-glorot_function(in_dim,hid_dim), high=glorot_function(in_dim,hid_dim),size=(in_dim,hid_dim))
    b = np.random.uniform(low=-glorot_function(in_dim,hid_dim), high=glorot_function(hid_dim,out_dim),size=hid_dim)
    U = np.random.uniform(low=-glorot_function(hid_dim,out_dim), high=glorot_function(hid_dim,out_dim),size=(hid_dim,out_dim))
    b_tag = np.random.uniform(low=-glorot_function(hid_dim,out_dim), high=glorot_function(hid_dim,out_dim),size=(out_dim))

    params = [W, b, U, b_tag]
    return params


if __name__ == '__main__':
    
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check
    
    W, b, U, b_tag = create_classifier(3, 4, 5)
    
    
    def _loss_and_W_grad(W):
        """
            This function creates random x vector, canculates on it loss and gradients for w and
            returns loss and w-gradients that were calculated for random x vector.
            """
        global b
        global U
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 3, [W, b, U, b_tag])
        return loss, grads[0]
    
    
    def _loss_and_b_grad(b):
        global W
        global U
        global b_tag
        loss, grads = loss_and_gradients([1, 2, 3], 3, [W, b, U, b_tag])
        return loss, grads[1]
    
    
    def _loss_and_U_grad(U):
        global W
        global b
        global b_tag
        loss, grads = loss_and_gradients([0, 0, 0 ], 2, [W, b, U, b_tag])
        return loss, grads[2]
    
    
    def _loss_and_b_tag_grad(b_tag):
        global W
        global b
        global U
        loss, grads = loss_and_gradients([1, 2, 100, -20, 3, 6, 7], 3, [W, b, U, b_tag])
        return loss, grads[3]
    
    
    for _ in range(1000):
        W = np.random.randn(W.shape[0], W.shape[1])
        b = np.random.randn(b.shape[0])
        U = np.random.randn(U.shape[0], U.shape[1])
# b_tag = np.random.randn(b_tag.shape[0])
# gradient_check(_loss_and_b_grad, b)
#gradient_check(_loss_and_W_grad, W)
#gradient_check(_loss_and_U_grad, U)
# gradient_check(_loss_and_b_tag_grad, b_tag)
