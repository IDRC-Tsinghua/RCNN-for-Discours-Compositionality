import numpy
import theano
import theano.tensor as T

class Average(object):

    def __init__(self, input, dim):
        """
        :type input: theano.tensor.matrix
        :param input: symbolic sentence tensor

        :type dim: int
        :param dim: dimensions of a word vector
        """
        self.input = input
        self.dim = dim
        self.params = []

        self.output = T.mean(self.input, axis=0)
