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

    def __init__(self, rng, input, length, dim, activation=T.tanh):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic sentence tensor, of shape (1, 1, length, dim)

        :type length: int
        :param length: maximum length of a sentence

        :type dim: int
        :param dim: dimensions of a word vector

        :type activation: theano.Op or function
        :param activation: None linearity to be applied in the hidden layer
        """
        W_bound = numpy.sqrt(6. / (length * dim))
        self.input = input

        # construct parameters
        self.params = list()
        self.layers = 0
        self.ns = list()
        self.ks = list()
        n = length
        k = 2
        while n > 1:
            filter_init = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound,
                size=(1, 1, k, 1)), dtype=theano.config.floatX)
            f = theano.shared(value=filter_init)
            n -= k - 1
            self.params.append(f)
            self.layers += 1
            self.ks.append(k)
            self.ns.append(n)
            k += 1
            if k > n:
                k = n
        b_init = numpy.zeros((1, 1, 1, dim), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_init)
        self.params.append(self.b)

        # construct hcnn
        self.nodes = list()
        self.nodes.append(self.input)
        for i in range(self.layers):
            n = length if i == 0 else self.ns[i - 1]
            hidden = conv.conv2d(input=self.nodes[i], filters=self.params[i],
                filter_shape=(1, 1, self.ks[i], 1), image_shape=(1, 1, n, dim))
            self.nodes.append(hidden)

        self.output = activation(self.nodes[self.layers] + self.b)
