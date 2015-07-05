from hcnn import HCNN
from rnn import RNN

import cPickle
import gzip
import logging

import numpy
import theano
import theano.tensor as T

class RCNN(object):
    """combination of hcnn and rnn
    """
    def __init__(self, rng, input, length, dim, n_hidden, n_out):
        hcnn_input = input.reshape((1, 1, length, dim))
        self.hcnn = HCNN(rng=rng, input=hcnn_input, length=length, dim=dim)
        hcnn_output = self.hcnn.output.reshape((1, dim))
        self.rnn = RNN(rng=rng, input=hcnn_output,
            n_in=dim, n_hidden=n_hidden, n_out=n_out)
        self.p_y_given_x = self.rnn.p_y_given_x
        self.loss = self.rnn.loss
        self.crossentropy = self.rnn.crossentropy
        self.errors = self.rnn.errors
        self.params = self.hcnn.params + self.rnn.params


def load_data(dataset):
    logging.info('Loading data ...')
    TRAIN_SET = 200000
    VALID_SET = 210000
    f = gzip.open(dataset, 'rb')
    data = cPickle.load(f)
    f.close()
    data_x, data_y = data
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    valid_x = data_x[TRAIN_SET:VALID_SET]
    valid_y = data_y[TRAIN_SET:VALID_SET]
    test_x = data_x[VALID_SET:]
    test_y = data_y[VALID_SET:]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def test_rcnn(length, dim, n_out, n_hidden, lr=0.01, n_epochs=1000,
    validation_frequency=10000, dataset='data/swda.pkl.gz'):
    logging.info(('Starting with length = %d, dim = %d, n_out = %d, n_hidden' +
        ' = %d, lr = %f, n_epochs = %d') % ( length, dim, n_out, n_hidden, lr,
        n_epochs))
    # load data
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(dataset)
    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)

    logging.info('Initializing ...')
    # Expand x to length with 0 vector.
    def expand_x(x):
        res = list()
        l = len(x)
        for i in xrange(length):
            if i < l:
                res.append(x[i])
            else:
                res.append([0]*dim)
        return numpy.array(res, dtype=theano.config.floatX)

    # Expand y to [..., 0, 1, 0, ...]
    def expand_y(y):
        res = [0] * n_out
        res[y] = 1
        return numpy.array(res, dtype=theano.config.floatX)

    # define variables
    x = T.matrix('x')
    y = T.vector('y')
    yi = T.scalar('yi')

    # define network
    rcnn = RCNN(rng=numpy.random.RandomState(12345), input=x, length=length,
        dim=dim, n_hidden=n_hidden, n_out=n_out)
    errors = rcnn.errors(yi)
    cost = rcnn.loss(y)

    # define functions
    compute_error = theano.function(inputs=[x, yi], outputs=errors)
    compute_loss = theano.function(inputs=[x, y], outputs=cost)

    gparams = [T.grad(cost, param) for param in rcnn.params]
    updates = [(param, param - lr * gparam) for (
        param, gparam) in zip(rcnn.params, gparams)]
    train_model = theano.function(inputs=[x, y], outputs=cost, updates=updates)

    # train
    logging.info('Training ...')
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for idx in xrange(n_train):
            train_model(expand_x(train_x[idx]), expand_y(train_y[idx]))
            iter = (epoch - 1) * n_train + idx + 1
            if iter % validation_frequency == 0:
                valid_errors = [compute_error(expand_x(valid_x[i]),
                    valid_y[i]) for i in xrange(n_valid)]
                mean_error = numpy.mean(valid_errors)
                valid_loss = [compute_loss(expand_x(valid_x[i]),
                    expand_y(valid_y[i])) for i in xrange(n_valid)]
                mean_loss = numpy.mean(valid_loss)
                logging.info(
                    'Epoch %i, seq %i/%i, valid error %f, valid loss %f' % (
                    epoch, idx+1, n_train, mean_error, mean_loss))

    test_errors = [compute_error(
        expand_x(test_x[i]), test_y[i]) for i in xrange(n_test)]
    mean_error = numpy.mean(test_errors)
    test_losses = [compute_loss(
        expand_x(test_x[i]), expand_y(test_y[i])) for i in xrange(n_test)]
    mean_loss = numpy.mean(test_losses)
    logging.info('Test error %f, test loss %f' % (mean_error, mean_loss))
    logging.info('Done')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%d %b %Y %H:%M:%S', filename='rcnn.log', filemode='w')
    test_rcnn(length=78, dim=25, n_out=217, n_hidden=400)
