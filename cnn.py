import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

class CNN(object):

    def __init__(self, rng, input, length, dim, n_feature_maps, window_sizes):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic sentence tensor, of shape (1, 1, length, dim)

        :type length: int
        :param length: maximum length of a sentence

        :type dim: int
        :param dim: dimensions of a word vector

        :type n_feature_maps: int
        :param n_feature_maps: number of feature maps

        :type window_sizes: tuple of int
        :param window_sizes: convolution kernel window sizes
        """
        W_bound = numpy.sqrt(6. / (length * dim))
        self.input = input
        self.length = length
        self.dim = dim
        self.n_feature_maps = n_feature_maps
        self.window_sizes = window_sizes

        self.params = []
        reshaped_input = self.input.dimshuffle('x', 'x', 0, 1)
        self.output = None
        for window_size in window_sizes:
            W_init = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound,
                size=(self.n_feature_maps, 1, window_size, self.dim)),
                dtype=theano.config.floatX)
            W = theano.shared(value=W_init)
            self.params.append(W)
            conv_out = conv.conv2d(input=reshaped_input, filters=W)
            max_out = T.max(conv_out, axis=2).flatten()
            self.output = max_out if self.output is None else T.concatenate(
                [self.output, max_out])
        b_init = numpy.zeros((self.n_feature_maps * len(self.window_sizes),),
            dtype=theano.config.floatX)
        self.b = theano.shared(value=b_init)
        self.params.append(self.b)
        self.output = self.output + self.b
