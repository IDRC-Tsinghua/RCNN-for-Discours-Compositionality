import cPickle as pickle
import gzip
import random
from math import sqrt
from operator import itemgetter
from collections import defaultdict
from swda import CorpusReader


vector_size = 300
swda_path = 'swda'
tags_file = 'swda.tags'
data_file = 'swda.random.pkl.gz'
except_words = set([',',])


def random_vector(size):
    res = [random.random() for i in xrange(size)]
    length = sqrt(sum([i * i for i in res]))
    return [i / length for i in res]


# convert each utterance into a list of word vectors(presented as list),
# convert tag into it's number. return a list with element formed like
# ([word_vec1, word_vec2, ...], tag_no)
def process_data(tags):
    x = []
    y = []
    model= {}
    corpus = CorpusReader(swda_path)
    for utt in corpus.iter_utterances():
        words = [w.lower() for w in utt.pos_words() if w not in except_words]
        for word in words:
            if word not in model:
                model[word] = random_vector(vector_size)
        words = [model[w] for w in words]
        tag = tags[utt.damsl_act_tag()]
        x.append(words)
        y.append(tag)
    return (x, y)


def save_data(data, pickle_file):
    f = gzip.GzipFile(pickle_file, 'w')
    pickle.dump(data, f)
    f.close()


# load corpus and save number of tags into tags_file.
def preprocess_data():
    act_tags = defaultdict(lambda: 0)
    corpus = CorpusReader(swda_path)
    for utt in corpus.iter_utterances():
        act_tags[utt.damsl_act_tag()] += 1
    act_tags = act_tags.iteritems()
    act_tags = sorted(act_tags, key=itemgetter(1), reverse=True)
    f = open(tags_file, 'w')
    for k, v in act_tags:
        f.write('%s %d\n' % (k, v))
    f.close()
    return dict([(act_tags[i][0], i) for i in xrange(len(act_tags))])


def main():
    print 'Preprocessing data ...'
    tags = preprocess_data()
    print 'Reading and converting data from swda ...'
    data = process_data(tags)
    print 'Saving ...'
    save_data(data, data_file)


if __name__ == '__main__':
    main()
