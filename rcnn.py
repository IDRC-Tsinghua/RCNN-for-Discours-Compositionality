from rnn import RNN
from cnn import CNN

import cPickle
import gzip
import logging

import numpy
import theano
import theano.tensor as T


class RCNN(object):
    """combination of cnn and rnn
    """
    def __init__(self, rng, input, h_prev, y_prev, length, dim, n_feature_maps,
            window_sizes, n_hidden, n_out):
        self.cnn = CNN(rng=rng, input=input, length=length, dim=dim,
            n_feature_maps=n_feature_maps, window_sizes=window_sizes)
        self.rnn = RNN(rng=rng, input=self.cnn.output, h_prev=h_prev,
            y_prev=y_prev, n_in=300, n_hidden=n_hidden, n_out=n_out)
        self.s = self.cnn.output
        self.h = self.rnn.h
        self.y = self.rnn.y
        self.output = self.rnn.output
        self.loss = self.rnn.loss
        self.error = self.rnn.error
        self.params = self.cnn.params + self.rnn.params


def load_data(dataset):
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


def test_rcnn(length, dim, n_out, n_feature_maps, window_sizes, n_hidden,
    lr=0.001, n_epochs=1000000, validation_frequency=1000,
    dataset='data/swda.pkl.gz'):

    logging.info('length = %d, dim = %d, n_out = %d' % (length, dim, n_out))
    logging.info('n_feature_maps = %d, window_sizes={}, n_hidden = %d'.format(
        window_sizes) % (n_feature_maps, n_hidden))
    logging.info('lr = %f, n_epochs = %d' % (lr, n_epochs))

    print 'Initializing ...'
    # define variables
    x = T.matrix('x')
    y = T.vector('y')
    label = T.scalar('label')
    h_prev = T.vector('h_prev')
    y_prev = T.vector('y_prev')

    # define network
    rcnn = RCNN(rng=numpy.random.RandomState(54321), input=x, h_prev=h_prev,
        y_prev=y_prev, length=length, dim=dim, n_feature_maps=n_feature_maps,
        window_sizes=window_sizes, n_hidden=n_hidden, n_out=n_out)
    h_cur = rcnn.h
    output = rcnn.output
    error = rcnn.error(label)
    cost = rcnn.loss(y)

    # define functions
    compute_error = theano.function(
        inputs=[x, h_prev, y_prev, label], outputs=[error, h_cur, output])
    compute_loss = theano.function(
        inputs=[x, h_prev, y_prev, y], outputs=[cost, h_cur, output, rcnn.s,
        rcnn.rnn.y])
    gparams = [T.grad(cost, param) for param in rcnn.params]
    updates = [(param, param - lr * gparam) for (
        param, gparam) in zip(rcnn.params, gparams)]
    train_model = theano.function(
        inputs=[x, h_prev, y_prev, y], outputs=h_cur, updates=updates)

    print 'Loading data ...'
    train_x, train_label, valid_x, valid_label, test_x, test_label = load_data(
        dataset)
    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    # Expand x to length with 0 vector.
    def expand_x(x, min_size=max(window_sizes)):
        fill_size = min_size - len(x)
        if fill_size > 0:
            x += [[0] * dim for i in xrange(fill_size)]
        x = numpy.array(x, dtype=theano.config.floatX)
        return x
    # Expand y to [..., 0, 1, 0, ...]
    def expand_y(y):
        res = [0] * n_out
        res[y] = 1
        return numpy.array(res, dtype=theano.config.floatX)
    train_x = [expand_x(i) for i in train_x]
    train_y = [expand_y(i) for i in train_label]
    valid_x = [expand_x(i) for i in valid_x]
    valid_y = [expand_y(i) for i in valid_label]
    test_x = [expand_x(i) for i in test_x]
    test_y = [expand_y(i) for i in test_label]

    print 'Training ...'
    h_prev_train = [0] * n_hidden
    y_prev_train = expand_y(0)
    h_prev_valid = [0] * n_hidden
    y_prev_valid = expand_y(0)
    h_prev_test = [0] * n_hidden
    y_prev_test = expand_y(0)
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for idx in xrange(n_train):
            # train
            h_prev_train = train_model(
                train_x[idx], h_prev_train, y_prev_train, train_y[idx])
            h_prev_train = h_prev_train.flatten()
            y_prev_train = train_y[idx]

            # valid
            iter = (epoch - 1) * n_train + idx + 1
            if iter % validation_frequency == 0:
                valid_errors = 0
                for idx_v in xrange(n_valid):
                    [valid_error, h_prev_valid, y_prev_valid] = compute_error(
                        valid_x[idx_v], h_prev_valid, y_prev_valid,
                        valid_label[idx_v])
                    h_prev_valid = h_prev_valid.flatten()
                    y_prev_valid = expand_y(y_prev_valid[0])
                    valid_errors += valid_error
                mean_error = valid_errors / n_valid
                valid_losses = 0
                print
                print '-----'
                for idx_v in xrange(n_valid):
                    [valid_loss, h_prev_valid, y_prev_valid,ss,yy]=compute_loss(
                        valid_x[idx_v], h_prev_valid, y_prev_valid,
                        valid_y[idx_v])
                    h_prev_valid = h_prev_valid.flatten()
                    """
                    if idx_v < 10: # debug info
                        print 's = '
                        print ss
                        print 'h = '
                        print h_prev_valid
                        print 'y = '
                        print yy
                        print valid_label[idx_v]
                    """
                    print y_prev_valid[0],
                    y_prev_valid = expand_y(y_prev_valid[0])
                    valid_losses += valid_loss
                mean_loss = valid_losses / n_valid
                logging.info(
                    'Epoch %i, seq %i/%i, valid error %f, valid loss %f' % (
                    epoch, idx+1, n_train, mean_error, mean_loss))
    #test
    test_errors = 0
    for idx in xrange(n_test):
        [test_error, h_prev_test, y_prev_test] = compute_error(
            test_x[idx], h_prev_test, y_prev_test, test_label[idx])
        h_prev_test = h_prev_test.flatten()
        y_prev_test = expand_y(y_prev_test[0])
        test_errors += test_error
    mean_error = test_errors / n_test
    test_losses = 0
    for idx in xrange(n_test):
        [test_loss, h_prev_test, y_prev_test] = compute_loss(
            test_x[idx], h_prev_test, y_prev_test, test_y[idx])
        h_prev_test = h_prev_test.flatten()
        y_prev_test = expand_y(y_prev_test[0])
        test_losses += test_loss
    mean_loss = test_losses / n_test
    logging.info(
        'Epoch %i, seq %i/%i, test error %f, test loss %f' % (
        epoch, idx+1, n_train, mean_error, mean_loss))
    print 'Done'

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%d %b %Y %H:%M:%S', filename='training.log', filemode='w')
    test_rcnn(length=78, dim=25, n_out=217, n_feature_maps=100,
        window_sizes=(3, 4, 5), n_hidden=200, lr=0.01)
