import csv
import os
import pickle
import nltk
import operator
import matplotlib.pyplot as plt


def analyse():

    plot_significant_words(pickle.load(open('pickles/bef_dict_true.p', 'rb')),
                           pickle.load(open('pickles/bef_dict_false.p', 'rb')), 'before')

    plot_significant_words(pickle.load(open('pickles/bet_dict_true.p', 'rb')),
                           pickle.load(open('pickles/bet_dict_false.p', 'rb')), 'between')

    plot_significant_words(pickle.load(open('pickles/aft_dict_true.p', 'rb')),
                           pickle.load(open('pickles/aft_dict_false.p', 'rb')), 'after')


def plot_significant_words(true_set, false_set, filename):
    """
    Plot graph showing words occurring significantly more often in true or false sets
    """
    basepath = os.path.dirname(__file__)
    filepath = os.path.abspath(os.path.join(basepath, 'plots', filename + '.tif'))

    true_set = sorted(true_set.iteritems(), key=operator.itemgetter(0), reverse=False)
    false_set = sorted(false_set.iteritems(), key=operator.itemgetter(0), reverse=False)

    # calculate difference by zipping together true and false examples and subtracting one from the other
    difference = [x - y for x, y in zip(zip(*true_set)[1], zip(*false_set)[1])]

    # zip differences back together with the terms
    dif = zip(zip(*true_set)[0], difference)
    # only interested in the significant terms
    dif = [d for d in dif if abs(d[1]) > 20]

    plt.bar(xrange(len(dif)), zip(*dif)[1])
    # using words as x axis labels
    plt.xticks(xrange(len(dif)), zip(*dif)[0], rotation='vertical', size='xx-small')
    plt.grid(True)
    plt.savefig(filepath, format='tif')
    # need to clear current figure or it will persist for next plot
    plt.clf()


def create_dictionary(stem=False):
    """
    Create dictionary for all terms in corpus
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    if stem:
        file_in = os.path.abspath(os.path.join(basepath, 'csv/tagged_sentences_stemmed.csv'))
    else:
        file_in = os.path.abspath(os.path.join(basepath, 'csv/tagged_sentences.csv'))

    true_dict, false_dict = [], []
    true_count, false_count = 0, 0
    stopwords = nltk.corpus.stopwords.words('english')

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            if eval(row['true_relation']):
                true_count += 1
                true_dict.extend([t[0] for t in eval(row['before_tags']) if t[0] not in stopwords])
                true_dict.extend([t[0] for t in eval(row['between_tags']) if t[0] not in stopwords])
                true_dict.extend([t[0] for t in eval(row['after_tags']) if t[0] not in stopwords])
            else:
                false_count += 1
                false_dict.extend([t[0] for t in eval(row['before_tags']) if t[0] not in stopwords])
                false_dict.extend([t[0] for t in eval(row['between_tags']) if t[0] not in stopwords])
                false_dict.extend([t[0] for t in eval(row['after_tags']) if t[0] not in stopwords])

    true_dict = set([t for t in true_dict if t not in stopwords])
    false_dict = set([t for t in false_dict if t not in stopwords])
    dictionary = true_dict.union(false_dict)

    pickle.dump(true_dict, open('pickles/true_dictionary.p', 'wb'))
    pickle.dump(false_dict, open('pickles/false_dictionary.p', 'wb'))
    pickle.dump(dictionary, open('pickles/dictionary.p', 'wb'))

    print true_count, false_count


def counting_dict(stem=False):
    """
    Count occurrences of words in before, between and after sections of sentence
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    if stem:
        file_in = os.path.abspath(os.path.join(basepath, 'csv/tagged_sentences_stemmed.csv'))
    else:
        file_in = os.path.abspath(os.path.join(basepath, 'csv/tagged_sentences.csv'))

    dictionary = pickle.load(open('pickles/dictionary.p', 'rb'))

    # each dictionary contains all words (no stopwords) and defaults counts of zero
    bef_dict = {'true': {w: 0 for w in dictionary}, 'false': {w: 0 for w in dictionary}}
    bet_dict = {'true': {w: 0 for w in dictionary}, 'false': {w: 0 for w in dictionary}}
    aft_dict = {'true': {w: 0 for w in dictionary}, 'false': {w: 0 for w in dictionary}}

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            true_rel = eval(row['true_relation'])
            # TODO divide by total number of documents
            add_to_dict(eval(row['before_tags']), bef_dict, true_rel)
            add_to_dict(eval(row['between_tags']), bet_dict, true_rel)
            add_to_dict(eval(row['after_tags']), aft_dict, true_rel)

    # pickle everything
    pickle.dump(normalise(bef_dict['true'], True), open('pickles/bef_dict_true.p', 'wb'))
    pickle.dump(normalise(bet_dict['true'], True), open('pickles/bet_dict_true.p', 'wb'))
    pickle.dump(normalise(aft_dict['true'], True), open('pickles/aft_dict_true.p', 'wb'))
    pickle.dump(normalise(bef_dict['false'], False), open('pickles/bef_dict_false.p', 'wb'))
    pickle.dump(normalise(bet_dict['false'], False), open('pickles/bet_dict_false.p', 'wb'))
    pickle.dump(normalise(aft_dict['false'], False), open('pickles/aft_dict_false.p', 'wb'))


def normalise(d, true_rel):
    """Normalise the word counts in dictionary based on number of sentences
    """
    if true_rel:
        factor = 639.0
    else:
        factor = 2252.0

    for k, v in d.items():
        d[k] = round(2252*v/factor, 3)

    return d


def add_to_dict(tags, w_dict, true_rel):
    """
    Add count of words to the dictionary
    """
    if true_rel:
        rel_key = 'true'
    else:
        rel_key = 'false'

    words = [t[0] for t in tags]
    for w in words:
        if w in w_dict[rel_key].keys():
            w_dict[rel_key][w] += 1


if __name__ == '__main__':
    #counting_dict(True)
    analyse()
    #create_dictionary(True)
