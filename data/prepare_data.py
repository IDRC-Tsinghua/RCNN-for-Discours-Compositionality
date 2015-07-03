import re
import cPickle as pickle
import multiprocessing
from operator import itemgetter
from collections import defaultdict
from swda import CorpusReader

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

vector_size = 25
swda_path = 'swda'
text_file = 'swda.text'
model_file = 'swda.model'
info_file = 'swda.info'
data_file = 'swda.pkl'


word_pattern = re.compile(r'[a-z\']+')
one_letter_word = ('i', 'a')


def str2wordlist(s):
    return [w for w in word_pattern.findall(s) if (
        len(w) > 1 or w in one_letter_word)]


# convert each utterance into a list of word vectors(numpy array), convert tag
# into it's # number. return a list whose element is formed as
# ([word_vec1, word_vec2, ...], tag_no)
def process_data(model, tags):
    print 'Reading and converting data from swda ...'
    x = list()
    y = list()
    corpus = CorpusReader(swda_path)
    for utt in corpus.iter_utterances():
        wordlist = str2wordlist(utt.text.lower())
        words = [model[w] for w in wordlist]
        tag = tags[utt.act_tag]
        x.append(words)
        y.append(tag)
    return (x, y)


def save_data(data, pickle_file):
    print 'Saving ...'
    f = open(pickle_file, 'w')
    pickle.dump(data, f)
    f.close()


# load corpus and save all the words into text_file as well as saving words
# info and number of tags into info_file.
def preprocess_data():
    print 'Preprocessing data ...'
    longest = 0
    act_tags = defaultdict(lambda: 0)
    corpus = CorpusReader(swda_path)
    f = open(text_file, 'w')
    for utt in corpus.iter_utterances():
        act_tags[utt.act_tag] += 1
        words = str2wordlist(utt.text.lower())
        longest = len(words) if len(words) > longest else longest
        f.write(' '.join(words))
        f.write(' ')
    f.close()
    act_tags = act_tags.iteritems()
    act_tags = sorted(act_tags, key=itemgetter(1), reverse=True)
    f = open(info_file, 'w')
    f.write('longest: %d\n' % longest)
    f.write('tags:\n')
    for k, v in act_tags:
        f.write('%s %d\n' % (k, v))
    f.close()
    return {act_tags[i][0]: i for i in xrange(len(act_tags))}


def train_word2vec(verbose=False):
    print 'Training word2vec ...'
    model = Word2Vec(LineSentence(text_file), size=vector_size, min_count=1,
            workers=multiprocessing.cpu_count())
    model.save(model_file)
    return model


def main():
    tags = preprocess_data()
    model = train_word2vec()
    data = process_data(model, tags)
    save_data(data, data_file)


if __name__ == '__main__':
    main()
