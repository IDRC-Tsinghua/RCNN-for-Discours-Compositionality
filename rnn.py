import numpy
import theano
import theano.tensor as T

class RNN(object):
    """Recurrent neural network
    """

    def __init__(self, rng, input, h_prev, y_prev, n_in, n_hidden, n_out,
        activation=T.tanh):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: symbolic tensor, of shape (1, n_in)

        :type h_prev: theano.tensor.dmatrix
        :param h_prev: symbolic tensor, of shape (1, n_hidden)

        :type y_prev: theano.tensor.dmatrix
        :param y_prev: symbolic tensor, of shape (1, n_out)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_hidden: int
        :param n_hidden: dimensionality of hidden layer

        :type n_out: int
        :param n_out: dimensionality of output

        :type activation: theano.Op or function
        :param activation: None linearity to be applied in the hidden layer
        """
        self.input = input
        self.h_prev = h_prev
        self.y_prev = y_prev
        self.activation = activation
        self.softmax = T.nnet.softmax

        W_init = numpy.asarray(rng.uniform(size=(n_hidden, n_hidden),
            low=-0.1, high=0.1), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init)

        W_in_init = numpy.asarray(rng.uniform(size=(n_in, n_hidden),
            low=-0.1, high=0.1), dtype=theano.config.floatX)
        self.W_in = theano.shared(value=W_in_init)

        W_out_init = numpy.asarray(rng.uniform(size=(n_hidden, n_out),
            low=-0.1, high=0.1), dtype=theano.config.floatX)
        self.W_out = theano.shared(value=W_out_init)

        W_prev_init = numpy.asarray(rng.uniform(size=(n_out, n_hidden),
            low=-0.1, high=0.1), dtype=theano.config.floatX)
        self.W_prev = theano.shared(value=W_prev_init)

        bh_init = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init)

        by_init = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init)

        self.params = [self.W, self.W_in, self.W_out, self.W_prev,
            self.bh, self.by]

        self.h = self.activation(T.dot(self.input, self.W_in) + T.dot(
            self.h_prev, self.W) + T.dot(self.y_prev, self.W_prev) + self.bh)
        self.y = self.softmax(T.dot(self.h, self.W_out) + self.by)

        self.output = T.argmax(self.y, axis=1)

    def loss(self, y):
        return T.mean(T.nnet.binary_crossentropy(self.y, y))

    def error(self, label):
        return T.mean(T.neq(self.output, label))
