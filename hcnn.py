import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

class HCNN(object):
    """Hierarchical convolutional neural network that composes a sentence which
    consists of 'length' word vectors, each of whom has 'dim' dimensions, into
    a 'dim'-dimensioned vector. (length, dim) -> (1, dim)
    The structure is convolutional network layers with kernel size k = 2, 3, ..
    and totally it has cell(sqrt(2 * length)) - 1 layers.
    """

    def __init__(self, rng, input, length, dim,
            sigmoid=T.nnet.sigmoid, activation=T.tanh):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic sentence tensor, of shape (1, 1, length, dim)

        :type length: int
        :param length: maximum length of a sentence

        :type dim: int
        :param dim: dimensions of a word vector

        :type sigmoid: theano.Op or function
        :param sigmoid: None linearity to be applied in the hidden layer

        :type activation: theano.Op or function
        :param activation: None linearity to be applied in the hidden layer
        """
        W_bound = numpy.sqrt(6. / (length * dim))
        self.input = input
        self.length = length
        self.dim = dim
        self.sigmoid = sigmoid
        self.activation = activation

        # construct parameters
        self.filters = []
        self.bs = []
        self.params = []
        self.layers = 0
        self.ns = []
        self.ks = []
        n = self.length
        k = 2
        while n > 1:
            n -= k - 1
            f_init = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound,
                size=(dim, k)), dtype=theano.config.floatX)
            f = theano.shared(value=f_init)
            self.filters.append(f)
            self.params.append(f)
            b_init = numpy.zeros((dim, n), dtype=theano.config.floatX)
            b = theano.shared(value=b_init)
            self.bs.append(b)
            self.params.append(b)
            self.layers += 1
            self.ks.append(k)
            self.ns.append(n)
            k += 1
            if k > n:
                k = n

        # construct hcnn
        def conv_on_row_maker(k, n):
            def conv_on_row(row, kernel):
                return conv.conv2d(
                    input=row.dimshuffle('x', 'x', 'x', 0),
                    filters=kernel.dimshuffle('x', 'x', 'x', 0),
                    filter_shape=(1, 1, 1, k),
                    image_shape=(1, 1, 1, n)).dimshuffle(3,)
            return conv_on_row
        self.nodes = []
        self.nodes.append(self.input.dimshuffle(1, 0))
        for i in xrange(self.layers):
            n = length if i == 0 else self.ns[i - 1]
            conved, updates = theano.scan(fn=conv_on_row_maker(self.ks[i], n),
                    outputs_info=None,
                    sequences=[self.nodes[i], self.filters[i]])
            hidden = self.sigmoid(conved + self.bs[i])
            self.nodes.append(hidden)

        #self.output = self.activation(self.nodes[self.layers]).dimshuffle(1,0)
        self.output = self.nodes[self.layers].dimshuffle(1, 0)
