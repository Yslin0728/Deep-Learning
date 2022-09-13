import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return outputs: Outputs, of the same shape as x
        :return cache: Cache, stored for backward computation, of the same shape as x
        """
        shape = x.shape
        outputs, cache = np.zeros(shape), np.zeros(shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Sigmoid activation function            #
        ########################################################################

        outputs = 1/(1+np.exp(-x))
        cache = outputs

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Sigmoid activation function           #
        ########################################################################
        
        dx = dout * cache * (1 - cache)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape

        :return out: Outputs, of the same shape as x
        :return cache: Cache, stored for backward computation, of the same shape as x
        """
        outputs = None
        cache = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Relu activation function               #
        ########################################################################
        
        outputs = np.maximum(0, x)
        cache = x
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return outputs, cache

    def backward(self, dout, cache):
        """
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None
        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass of Relu activation function              #
        ########################################################################

        dx = dout*np.int64(cache > 0) #if>0 then*1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dx


def affine_forward(x, w, b):
    """
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M) (6,5)
    :param b: A numpy array of biases, of shape (M,) (5,)
    :return out: output, of shape (N, M) (10,5)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))
    ########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.    #
    # You will need to reshape the input into rows.                        #
    ########################################################################
    x_new = x.reshape(N, -1) #x new:(10,6)
    out = x_new.dot(w) + b #out:(10,5)
    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    # Hint: Don't forget to average the gradients dw and db                #
    ########################################################################

    N = x.shape[0]
    dx = dout.dot(w.T).reshape(x.shape) 
    dw = x.reshape(N, -1).T.dot(dout)/N
    db = np.sum(dout, axis=0)/N

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx, dw, db