import re
import cPickle as pickle
import gzip
import numpy
from operator import itemgetter
from collections import defaultdict
from swda import CorpusReader
from gensim.models import Word2Vec


swda_path = 'swda'
model_file = 'vectors.bin.gz'
tags_file = 'swda.tags'
data_file = 'swda.pretrain.pkl.gz'

word_pattern = re.compile(r'[a-z\']+')
except_words = ('and', 'of', 'to')
accept_words = ('i',)


def str2wordlist(s):
    words = [w.split('\'')[0] for w in word_pattern.findall(s)]
    return [w for w in words if (
        len(w) > 1 and w not in except_words or w in accept_words)]


# convert each utterance into a list of word vectors(presented as list),
# convert tag into it's number. return a list with element formed like
# ([word_vec1, word_vec2, ...], tag_no)
def process_data(model, tags):
    x = []
    y = []
    model_cache = {}
    non_modeled = set()
    corpus = CorpusReader(swda_path)
    for utt in corpus.iter_utterances():
        wordlist = str2wordlist(utt.text.lower())
        for word in wordlist:
            if word in model:
                if word not in model_cache:
                    model_cache[word] = model[word].tolist()
            else:
                non_modeled.add(word)
        words = [model_cache[w] for w in wordlist if w in model_cache]
        tag = tags[utt.damsl_act_tag()]
        x.append(words)
        y.append(tag)
    print 'Complete. The following words are not converted: '
    print list(non_modeled)
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
    print 'Loading model ...'
    model = Word2Vec.load_word2vec_format(model_file, binary=True)
    print 'Reading and converting data from swda ...'
    data = process_data(model, tags)
    print 'Saving ...'
    save_data(data, data_file)


if __name__ == '__main__':
    main()
