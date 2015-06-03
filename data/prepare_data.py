import re
import cPickle as pickle
import word2vec
from operator import itemgetter
from collections import defaultdict
from swda import CorpusReader


vector_size = 10
utt_per_pickle = 50000
swda_path = 'swda'
text_file = 'swda.text'
model_file = 'swda.bin'
info_file = 'swda.info'
pickle_file_template = 'swda.{}.pkl'
word_pattern = re.compile(r'[a-z]+')


def read_data(tags, longest_utt):
    print 'Reading data from swda ...'
    model = word2vec.load(model_file)
    data = list()
    corpus = CorpusReader(swda_path)
    count = 0
    for utt in corpus.iter_utterances():
        # TODO A lot of words are missing from word2vec model and donno why
        # for w in word_pattern.findall(utt.text.lower()):
        #     if w not in model:
        #         print 'Warning: word "%s" not in word list.' % w
        words = [model[w] for w in word_pattern.findall(
            utt.text.lower()) if w in model]
        words = [words[i] if i < len(words) else [
            0 for j in range(vector_size)] for i in range(longest_utt)]
        tag = tags[utt.act_tag]
        data.append((words, tag))
        count += 1
        if counter % utt_per_pickle == 0:
            print data[0]
            save_data(
                data, pickle_file_template.format( counter / utt_per_pickle))
            data = list()


def save_data(data, pickle_file):
    f = open(pickle_file, 'w')
    pickle.dump(data, f)
    f.close()


def preprocess_data():
    print 'Proprocessing data ...'
    longest = 0
    act_tags = defaultdict(lambda: 0)
    corpus = CorpusReader(swda_path)
    f = open(text_file, 'w')
    for utt in corpus.iter_utterances():
        act_tags[utt.act_tag] += 1
        words = word_pattern.findall(utt.text.lower())
        longest = len(words) if len(words) > longest else longest
        f.write(' '.join(words))
        f.write('\n')
    f.close()
    act_tags = act_tags.iteritems()
    act_tags = sorted(act_tags, key=itemgetter(1), reverse=True)
    f = open(info_file, 'w')
    f.write('longest = %d\n' % longest)
    f.write('tags: \n')
    for k, v  in act_tags:
        f.write('%s %d\n' % (k ,v))
    f.close()
    return {act_tags[i][0]: i for i in range(len(act_tags))}, longest


def train_word2vec(verbose=False):
    print 'Training word2vec ...'
    word2vec.word2vec(
        text_file, model_file, size=vector_size, verbose=verbose)


def main():
    tags, longest_utt = preprocess_data()
    train_word2vec(verbose=False)
    read_data(tags, longest_utt)


if __name__ == '__main__':
    main()
