from rnn import RNN
from cnn import CNN
from avg import Average

import sys
import cPickle as pickle
import gzip
import logging
import numpy
import theano
import theano.tensor as T
from collections import defaultdict


class RCNN(object):
    """combination of cnn and rnn
    """
    def __init__(self, rng, input, h_prev, y_prev, dim, n_feature_maps,
            window_sizes, n_hidden, n_out):
        self.cnn = CNN(rng=rng, input=input, dim=dim,
            n_feature_maps=n_feature_maps, window_sizes=window_sizes)
        self.rnn = RNN(rng=rng, input=self.cnn.output, h_prev=h_prev,
            y_prev=y_prev, n_in=n_feature_maps*len(window_sizes),
            n_hidden=n_hidden, n_out=n_out)
        #self.avg = Average(input=input, dim=dim)
        #self.rnn = RNN(rng=rng, input=self.avg.output, h_prev=h_prev,
            #y_prev=y_prev, n_in=dim, n_hidden=n_hidden, n_out=n_out)
        self.h = self.rnn.h
        self.y = self.rnn.y
        self.output = self.rnn.output
        self.loss = self.rnn.loss
        self.error = self.rnn.error
        self.params = self.cnn.params + self.rnn.params
        #self.params = self.rnn.params


def load_data(dataset):
    TRAIN_SET = 200000
    VALID_SET = 210000
    f = gzip.open(dataset, 'rb')
    data = pickle.load(f)
    f.close()
    data_x, data_y = data
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    valid_x = data_x[TRAIN_SET:VALID_SET]
    valid_y = data_y[TRAIN_SET:VALID_SET]
    test_x = data_x[VALID_SET:]
    test_y = data_y[VALID_SET:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def test_rcnn(dim, n_out, n_feature_maps, window_sizes, n_hidden,
    lr=0.01, lr_decay=0.1, n_epochs=1, validation_frequency=1000,
    dataset='data/swda.pkl.gz'):

    logging.info('dim = %d, n_out = %d' % (dim, n_out))
    logging.info('n_feature_maps = %d, window_sizes={}, n_hidden = %d'.format(
        window_sizes) % (n_feature_maps, n_hidden))
    logging.info('lr = %f, lr_decay = %f, n_epochs = %d' % (
        lr, lr_decay, n_epochs))

    print 'Initializing ...'
    # define variables
    x = T.matrix('x')
    y = T.vector('y')
    label = T.scalar('label')
    h_prev = T.vector('h_prev')
    y_prev = T.vector('y_prev')
    learning_rate = T.scalar('learning_rate')

    # define network
    rcnn = RCNN(rng=numpy.random.RandomState(54321), input=x, h_prev=h_prev,
        y_prev=y_prev, dim=dim, n_feature_maps=n_feature_maps,
        window_sizes=window_sizes, n_hidden=n_hidden, n_out=n_out)
    h_cur = rcnn.h
    output = rcnn.output
    error = rcnn.error(label)
    cost = rcnn.loss(y)

    # define functions
    compute_error = theano.function(
        inputs=[x, h_prev, y_prev, label], outputs=[error, h_cur, output])
    compute_loss = theano.function(
        inputs=[x, h_prev, y_prev, y], outputs=[cost, h_cur, output])
    gparams = [T.grad(cost, param) for param in rcnn.params]
    updates = [(param, param - learning_rate * gparam) for (
        param, gparam) in zip(rcnn.params, gparams)]
    train_model = theano.function(inputs=[x, h_prev, y_prev, y, learning_rate],
        outputs=h_cur, updates=updates)

    print 'Loading data ...'
    train_x, train_label, valid_x, valid_label, test_x, test_label = load_data(
        dataset)
    n_train = len(train_x)
    n_valid = len(valid_x)
    n_test = len(test_x)
    # Expand x to max window size
    def expand_x(x, min_size=max(window_sizes)):
        fill_size = min_size - len(x)
        if fill_size > 0:
            x += [[0] * dim] * fill_size
        return x
    def wrap_x(x):
        return numpy.array(x, dtype=theano.config.floatX)
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
            iter = (epoch - 1) * n_train + idx
            # train
            h_prev_train = train_model(
                wrap_x(train_x[idx]), h_prev_train, y_prev_train, train_y[idx],
                lr * (lr_decay ** (float(iter) / n_train)))
            h_prev_train = h_prev_train.flatten()
            y_prev_train = train_y[idx]

            # valid
            if (iter + 1) % validation_frequency == 0:
                valid_errors = 0
                for idx_v in xrange(n_valid):
                    [valid_error, h_prev_valid, y_prev_valid] = compute_error(
                        wrap_x(valid_x[idx_v]), h_prev_valid, y_prev_valid,
                        valid_label[idx_v])
                    h_prev_valid = h_prev_valid.flatten()
                    y_prev_valid = expand_y(y_prev_valid[0])
                    valid_errors += valid_error
                mean_error = valid_errors / n_valid
                valid_losses = 0
                print '-----'
                n_pred = defaultdict(lambda: [0, 0, 0])
                for idx_v in xrange(n_valid):
                    [valid_loss, h_prev_valid, y_prev_valid] = compute_loss(
                        wrap_x(valid_x[idx_v]), h_prev_valid, y_prev_valid,
                        valid_y[idx_v])
                    h_prev_valid = h_prev_valid.flatten()
                    n_pred[y_prev_valid[0]][0] += 1 # for debug
                    n_pred[valid_label[idx_v]][2] += 1 # for debug
                    if y_prev_valid[0] == valid_label[idx_v]:
                        n_pred[y_prev_valid[0]][1] += 1 # for debug
                    y_prev_valid = expand_y(y_prev_valid[0])
                    valid_losses += valid_loss
                for k, v in n_pred.iteritems():
                    print '%d:\t%d\t%d/%d' % (k, v[0], v[1], v[2])
                mean_loss = valid_losses / n_valid
                logging.info(
                    'Epoch %i, seq %i/%i, valid error %f, valid loss %f' % (
                    epoch, idx+1, n_train, mean_error, mean_loss))
    #test
    test_errors = 0
    for idx in xrange(n_test):
        [test_error, h_prev_test, y_prev_test] = compute_error(
            wrap_x(test_x[idx]), h_prev_test, y_prev_test, test_label[idx])
        h_prev_test = h_prev_test.flatten()
        y_prev_test = expand_y(y_prev_test[0])
        test_errors += test_error
    mean_error = test_errors / n_test
    test_losses = 0
    for idx in xrange(n_test):
        [test_loss, h_prev_test, y_prev_test] = compute_loss(
            wrap_x(test_x[idx]), h_prev_test, y_prev_test, test_y[idx])
        h_prev_test = h_prev_test.flatten()
        y_prev_test = expand_y(y_prev_test[0])
        test_losses += test_loss
    mean_loss = test_losses / n_test
    logging.info('Test error %f, test loss %f' % (mean_error, mean_loss))
    print 'Done'

def main(log_file):
    logging.basicConfig(
        level=logging.DEBUG, format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', filename=log_file, filemode='w')
    test_rcnn(dim=300, n_out=43, n_feature_maps=100, window_sizes=(2, 3, 4),
        n_hidden=500, lr=0.02, lr_decay=0.01, dataset='data/swda.random.pkl.gz')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: python %s <log_file>' % sys.argv[0]
    else:
        main(sys.argv[1])
