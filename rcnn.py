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
    def __init__(self, rng, input, h_prev, y_prev, length, dim, n_hidden,
        n_out):
        self.hcnn = HCNN(rng=rng, input=input, length=length, dim=dim)
        self.rnn = RNN(rng=rng, input=self.hcnn.output, h_prev=h_prev,
            y_prev=y_prev, n_in=dim, n_hidden=n_hidden, n_out=n_out)
        self.s = self.hcnn.output
        self.h = self.rnn.h
        self.y = self.rnn.y
        self.output = self.rnn.output
        self.loss = self.rnn.loss
        self.error = self.rnn.error
        self.params = self.hcnn.params + self.rnn.params


def load_data(dataset):
    """
    train_x = [
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            ]
    train_y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    valid_x = [
            [[1.0, 0.0], [0.0, 1.0]],#ab
            [[1.0, 0.0], [0.0, 1.0]],#ab
            [[0.0, 1.0]],#b
            [[1.0, 0.0]],#a
            [[1.0, 0.0], [0.0, 1.0]],#ab
            [[1.0, 0.0]],#a
            [[0.0, 1.0]],#b
            [[0.0, 1.0]],#b
            [[0.0, 1.0], [0.0, 1.0]],#bb
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[0.0, 1.0]],#b
            [[1.0, 0.0], [0.0, 1.0]],#ab
            [[1.0, 0.0]],#a
            [[1.0, 0.0]],#a
            [[1.0, 0.0], [1.0, 0.0]],#aa
            ]
    valid_y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return train_x, train_y, valid_x, valid_y, [], []
    """
    print 'Loading data ...'
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


def test_rcnn(length, dim, n_out, n_hidden, lr=0.001, n_epochs=1000000,
    validation_frequency=1000, dataset='data/swda.pkl.gz'):

    logging.info(('Starting with length = %d, dim = %d, n_out = %d, n_hidden' +
        ' = %d, lr = %f, n_epochs = %d') % ( length, dim, n_out, n_hidden, lr,
        n_epochs))

    # load data
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(dataset)
    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)

    print 'Initializing ...'
    # Expand x to length with 0 vector.
    def expand_x(x):
        res = []
        l = len(x)
        for i in xrange(length):
            if i < l:
                res.append(x[i])
            else:
                res.append([0] * dim)
        return numpy.array(res, dtype=theano.config.floatX)

    # Expand y to [..., 0, 1, 0, ...]
    def expand_y(y):
        res = [0] * n_out
        res[y] = 1
        return numpy.array(res, dtype=theano.config.floatX)

    # define variables
    x = T.matrix('x')
    y = T.vector('y')
    label = T.scalar('label')
    h_prev = T.vector('h_prev')
    y_prev = T.vector('y_prev')

    # define network
    rcnn = RCNN(rng=numpy.random.RandomState(54321), input=x, h_prev=h_prev,
        y_prev=y_prev, length=length, dim=dim, n_hidden=n_hidden, n_out=n_out)
    h_cur = rcnn.h
    output = rcnn.output
    error = rcnn.error(label)
    cost = rcnn.loss(y)

    # define functions
    compute_error = theano.function(
        inputs=[x, h_prev, y_prev, label], outputs=[error, h_cur, output])
    compute_loss = theano.function(
        inputs=[x, h_prev, y_prev, y], outputs=[cost, h_cur, output, rcnn.s])

    gparams = [T.grad(cost, param) for param in rcnn.params]
    updates = [(param, param - lr * gparam) for (
        param, gparam) in zip(rcnn.params, gparams)]
    train_model = theano.function(
        inputs=[x, h_prev, y_prev, y], outputs=h_cur, updates=updates)

    # train
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
            h_prev_train = train_model(expand_x(train_x[idx]), h_prev_train,
                y_prev_train, expand_y(train_y[idx]))
            h_prev_train = h_prev_train.ravel()
            y_prev_train = expand_y(train_y[idx])

            # valid
            iter = (epoch - 1) * n_train + idx + 1
            if iter % validation_frequency == 0:
                valid_errors = 0
                for idx_v in xrange(n_valid):
                    [valid_error, h_prev_valid, y_prev_valid] = compute_error(
                        expand_x(valid_x[idx_v]), h_prev_valid, y_prev_valid,
                        valid_y[idx_v])
                    #print y_prev_valid[0],
                    h_prev_valid = h_prev_valid.ravel()
                    y_prev_valid = expand_y(y_prev_valid[0])
                    valid_errors += valid_error
                #print
                mean_error = valid_errors / n_valid
                valid_losses = 0
                for idx_v in xrange(n_valid):
                    [valid_loss, h_prev_valid, y_prev_valid, s] = compute_loss(
                        expand_x(valid_x[idx_v]), h_prev_valid, y_prev_valid,
                        expand_y(valid_y[idx_v]))
                    h_prev_valid = h_prev_valid.ravel()
                    y_prev_valid = expand_y(y_prev_valid[0])
                    valid_losses += valid_loss
                print s
                #print rcnn.rnn.W_in.get_value()
                #print rcnn.rnn.W_prev.get_value()
                #print rcnn.rnn.W.get_value()
                #print rcnn.rnn.bh.get_value()
                #print rcnn.rnn.W_out.get_value()
                #print rcnn.rnn.by.get_value()
                mean_loss = valid_losses / n_valid
                logging.info(
                    'Epoch %i, seq %i/%i, valid error %f, valid loss %f' % (
                    epoch, idx+1, n_train, mean_error, mean_loss))
    #test
    test_errors = 0
    for idx in xrange(n_test):
        [test_error, h_prev_test, y_prev_test] = compute_error(
            expand_x(test_x[idx]), h_prev_test, y_prev_test,
            test_y[idx])
        h_prev_test = h_prev_test.ravel()
        y_prev_test = expand_y(y_prev_test[0])
        test_errors += test_error
    mean_error = test_errors / n_test
    test_losses = 0
    for idx in xrange(n_test):
        [test_loss, h_prev_test, y_prev_test, s] = compute_loss(
            expand_x(test_x[idx]), h_prev_test, y_prev_test,
            expand_y(test_y[idx]))
        h_prev_test = h_prev_test.ravel()
        y_prev_test = expand_y(y_prev_test[0])
        test_losses += test_loss
    mean_loss = test_losses / n_test
    logging.info(
        'Epoch %i, seq %i/%i, test error %f, test loss %f' % (
        epoch, idx+1, n_train, mean_error, mean_loss))
    print 'Done'

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%d %b %Y %H:%M:%S', filename='training.log', filemode='w')
    test_rcnn(length=78, dim=25, n_out=217, n_hidden=100, lr=0.1)
    #test_rcnn(length=2, dim=2, n_out=2, n_hidden=2, lr=0.1)
