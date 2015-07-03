import numpy
import theano
import theano.tensor as T

class RNN(object):
    """Recurrent neural network
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, activation=T.tanh):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: symbolic tensor, of shape (1, n_in)

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

        W_back_init = numpy.asarray(rng.uniform(size=(n_out, n_hidden),
            low=-0.1, high=0.1), dtype=theano.config.floatX)
        self.W_back = theano.shared(value=W_back_init)

        h0_init = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init)
        y0_init = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.y0 = theano.shared(value=y0_init)

        bh_init = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init)

        by_init = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init)

        self.params = [self.W, self.W_in, self.W_out,
            self.h0, self.bh, self.by]

        def step(x_t, h_tm1, y_tm1):
            h_t = self.activation(T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W)
                + T.dot(y_tm1, self.W_back)+ self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t
        [self.h, self.y_pred], up = theano.scan(fn=step, sequences=self.input,
            outputs_info=[self.h0, self.y0])

        self.p_y_given_x = self.softmax(self.y_pred)
        self.y_out = T.argmax(self.p_y_given_x, axis=-1)
        self.loss = lambda y: self.crossentropy(y)

    def crossentropy(self, y):
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def errors(self, yi):
        return T.mean(T.neq(self.y_out, yi))
