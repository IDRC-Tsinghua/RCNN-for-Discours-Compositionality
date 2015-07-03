from hcnn import HCNN
from rnn import RNN

import cPickle
import gzip

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
    print 'Loading data ...'
    f = gzip.open(dataset, 'rb')
    data = cPickle.load(f)
    f.close()
    data_x, data_y = data
    train_x = data_x[:10000]
    train_y = data_y[:10000]
    test_x = data_x[10000:11000]
    test_y = data_y[10000:11000]

    return train_x, train_y, test_x, test_y


def test_rcnn(length, dim, n_out, n_hidden, lr=0.005, n_epochs=1000,
        validation_frequency=1000, dataset='data/swda.pkl.gz'):
    train_x, train_y, test_x, test_y = load_data(dataset)
    n_train = len(train_x)
    n_test = len(test_x)

    def expand_x(x):
        res = list()
        l = len(x)
        for i in xrange(length):
            if i < l:
                res.append(x[i])
            else:
                res.append([0]*dim)
        return numpy.array(res, dtype=theano.config.floatX)
    def expand_y(y):
        res = [0] * n_out
        res[y] = 1
        return numpy.array(res, dtype=theano.config.floatX)

    xx = T.matrix('xx')
    yy = T.vector('yy')
    ii = T.scalar('ii')
    x = T.matrix('x')
    y = T.vector('y')
    yi = T.scalar('yi')

    rcnn = RCNN(rng=numpy.random.RandomState(12345), input=x, length=length,
        dim=dim, n_hidden=n_hidden, n_out=n_out)

    cost = rcnn.loss(y)
    compute_train_error = theano.function(inputs=[xx, ii],
        outputs=rcnn.errors(yi), givens={x: xx, yi: ii})
    compute_train_loss = theano.function(inputs=[xx, yy],
        outputs=cost, givens={x: xx, y: yy})
    compute_test_error = theano.function(inputs=[xx, ii],
        outputs=rcnn.errors(yi), givens={x: xx, yi: ii})
    compute_test_loss = theano.function(inputs=[xx, yy],
        outputs=cost, givens={x: xx, y: yy})

    gparams = []
    for param in rcnn.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []
    for param, gparam in zip(rcnn.params, gparams):
        updates.append((param, param - lr * gparam))

    train_model = theano.function(inputs=[xx, yy], outputs=cost,
        updates=updates, givens={x: xx, y: yy})

    print 'Training ...'
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        for idx in xrange(n_train):
            example_cost = train_model(expand_x(train_x[idx]),
                expand_y(train_y[idx]))
            iter = (epoch - 1) * n_train + idx + 1
            if iter % validation_frequency == 0:
                train_errors = [compute_train_error(expand_x(train_x[i]),
                    train_y[i]) for i in xrange(9000, n_train)]
                this_train_error = numpy.mean(train_errors)
                train_loss = [compute_train_loss(expand_x(train_x[i]),
                    expand_y(train_y[i])) for i in xrange(9000, n_train)]
                this_train_loss = numpy.mean(train_loss)
                test_errors = [compute_test_error(expand_x(test_x[i]),
                    test_y[i]) for i in xrange(n_test)]
                this_test_error = numpy.mean(test_errors)
                test_losses = [compute_test_loss(expand_x(test_x[i]),
                    expand_y(test_y[i])) for i in xrange(n_test)]
                this_test_loss = numpy.mean(test_losses)
                print ('epoch %i, seq %i/%i, train error %f, test error %f ' +
                    'train loss %f, test loss %f') % (epoch, idx+1, n_train,
                        this_train_error, this_test_error, this_train_loss,
                        this_test_loss)
    print 'Done'

if __name__ == '__main__':
    test_rcnn(length=78, dim=25, n_out=217, n_hidden=400)
