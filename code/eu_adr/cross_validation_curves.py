import sqlite3
import operator
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped

vec = DictVectorizer()
# need to specify database to use here
db_path = 'database/relex.db'


def load_records(which_set):
    """
    Load original and new data sets
    """
    with sqlite3.connect(db_path) as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # all records from original corpus
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')

        orig = cursor.fetchall()

        # now all newly annotated records
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source = 'pubmed' AND
                                true_rel IS NOT NULL;''')

        new = cursor.fetchall()

    # return whichver set was requested
    if which_set == 'original':
        return orig
    elif which_set == 'new':
        return new
    else:
        return orig, new


def build_pipeline(bag_of_words):
    """
    Set up classfier and extractor here to avoid repetition
    """
    if bag_of_words:
        clf = Pipeline([('normaliser', preprocessing.Normalizer(norm='l2')),
                        ('svm', LinearSVC(dual=True, C=1))])
        # set up extractor using desired features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, pos=False, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=True, bigrams=True)
        sim = pickle.load(open('pickles/cross_valid_similarities_orig_bag_of_words.p', 'rb'))

    else:
        clf = Pipeline([('normaliser', preprocessing.Normalizer(norm='l2')),
                        ('svm', SVC(kernel='rbf', gamma=100, cache_size=2000, C=10))])
        # set up extractor using desired features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, pos=True, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=False, bigrams=False)
        # below must be used to create active features
        #extractor.create_dictionaries(all_records, how_many=5)
        sim = pickle.load(open('pickles/cross_valid_similarities_orig_features_only.p', 'rb'))

    return clf, extractor, sim


def get_similarities(records, extractor):
    """
    Calculate similarities of vectors ie one to all others
    """
    data, _ = extractor.generate_features(records)
    data = vec.fit_transform(data).toarray()

    print 'calculating similarities'
    similarities = np.zeros(len(data))
    for i, v in enumerate(data):
        print i
        total = 0
        others = np.delete(data, i, 0)

        # loop through all other vectors and get total cosine distance
        for x in others:
            total += distance.cosine(v, x)

        # cos_similarity = 1 - av_cos_dist
        similarities[i] = 1 - total/len(others)

    return similarities


def pickle_similarities(which_set, extractor=None):
    """
    Pickle similarities based on all records
    """
    if which_set:
        records = load_records(which_set)
        # shuffle the records to mix up eu-adr and biotext
    else:
        # want to keep records separate for now in this case
        orig, new = load_records(which_set)

    if not extractor:
        # set up extractor using desired features
        '''
        # FOR THE SPARSE LINEAR SVM
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, pos=False, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=True, bigrams=True)
        '''
        # FOR THE FEATURES ONE
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, pos=True, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=False, bigrams=False)

    similarities = get_similarities(records, extractor)

    #pickle.dump(similarities, open('pickles/cross_valid_similarities_orig_bag_of_words.p', 'wb'))
    pickle.dump(similarities, open('pickles/cross_valid_similarities_orig_features_only.p', 'wb'))


def random_sampling(clf, data, labels, sets, splits, seed):
    """
    Calculate scores for random sampling
    """
    # set up arrays to hold scores
    accuracy = np.zeros(splits)
    precision = np.zeros(splits)
    recall = np.zeros(splits)
    fscore = np.zeros(splits)

    # initialise as empty array
    cur_indices = np.array([], dtype=int)

    # random can just use cross validation folds passed in
    # here we take the test split for training since we will be adding incrementally
    for i, (_, next_split) in enumerate(sets):
        # add next split to existing data set
        cur_indices = np.append(cur_indices, next_split)
        print len(cur_indices)
        cur_data = data[cur_indices]
        cur_labels = labels[cur_indices]

        # set up cross validator for this split
        cv = cross_validation.StratifiedKFold(cur_labels, shuffle=True, n_folds=10, random_state=seed*i)

        # add all scores for this split to array
        accuracy[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                               scoring='accuracy', n_jobs=-1))
        precision[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                                scoring='precision', n_jobs=-1))
        recall[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                             scoring='recall', n_jobs=-1))
        fscore[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                             scoring='f1', n_jobs=-1))

    return accuracy, precision, recall, fscore


def uncertainty_sampling(clf, data, labels, sets, splits, seed, similarities=None):
    """
    Calculate scores based on uncertainty sampling
    OR
    Calculate scores based on information density sampling if similarities parameter is passed
    """
    # set up arrays to hold scores
    accuracy = np.zeros(splits)
    precision = np.zeros(splits)
    recall = np.zeros(splits)
    fscore = np.zeros(splits)

    # initialise as empty array
    cur_indices = np.array([], dtype=int)
    all_indices = np.arange(len(data))
    # use same first set as other methods, sets is an iterable
    for _, next_split in sets:
        break

    # base number of samples to use on first split passed in to ensure matches other methods
    no_samples = len(data)/splits

    # random can just use cross validation folds passed in
    # here we take the test split for training since we will be adding incrementally
    for i in xrange(splits):
        # add next split to existing data set
        cur_indices = np.append(cur_indices, next_split)
        print len(cur_indices)
        cur_data = data[cur_indices]
        cur_labels = labels[cur_indices]

        # set up cross validator for this split
        cv = cross_validation.StratifiedKFold(cur_labels, shuffle=True, n_folds=10, random_state=seed*i)

        # add all scores for this split to array
        accuracy[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                               scoring='accuracy', n_jobs=-1))
        precision[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                                scoring='precision', n_jobs=-1))
        recall[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                             scoring='recall', n_jobs=-1))
        fscore[i] = np.mean(cross_validation.cross_val_score(clf, cur_data, cur_labels, cv=cv,
                                                             scoring='f1', n_jobs=-1))

        # now generate next split to add with uncertainty sampling methods
        # remove indices that have been used so far to generate remaining
        rest_indices = np.delete(all_indices, cur_indices)
        # don't try it on last split, will just fail
        if len(rest_indices) > 0:
            rest_data = data[rest_indices]

            # use decision function to find distance of each point from separating hyperplane
            clf.fit(cur_data, cur_labels)
            dist = clf.decision_function(rest_data).flatten()
            dist = np.absolute(dist)

            # if using density sampling divide by similarity
            # so distance is increased for less similar points
            if similarities is not None:
                rest_sim = similarities[np.array(rest_indices)]
                dist = np.divide(dist, rest_sim)
                # the beta parameter for information density sampling can be set here
                #rest_sim **= 0.8

            # zip it all together, order by distance then unzip
            remaining = sorted(zip(dist, rest_indices), key=operator.itemgetter(0))
            dist, rest_indices = zip(*remaining)
            # take those samples closest to hyperplane ie those most uncertain about
            next_split = rest_indices[:no_samples]

    return accuracy, precision, recall, fscore


def draw_learning_comparison(splits, r_score, u_score, d_score, samples_per_split, scoring):
    """
    Plot the different learning methods on same graph
    """
    # create ticks for x axis
    ticks = np.linspace(samples_per_split, splits*samples_per_split, splits)

    # set up the figure
    plt.figure()
    plt.grid()
    plt.xlabel('Training Instances')
    plt.ylabel(scoring)
    plt.title('Cross validation %s comparison using %s batches' % (scoring, splits))

    plt.plot(ticks, r_score, label='Random Sampling')
    plt.plot(ticks, u_score, label='Uncertainty Sampling')
    plt.plot(ticks, d_score, label='Density Sampling')

    plt.legend(loc='best')

    plt.savefig('plots/crossvalidation_learning_comparison_' + scoring + '_' + time_stamped('.png'), format='png')
    plt.clf()


def learning_comparison(splits, seed, which_set=None, bag_of_words=False):
    """
    Compare random, uncertainty and information density approaches
    """
    # seed it here so experiment is repeatable
    random.seed(0)

    # first load the data
    if which_set:
        records = load_records(which_set)
        # shuffle the records to mix up eu-adr and biotext
    else:
        # want to keep records separate for now in this case
        orig, new = load_records(which_set)
        len_orig = len(orig)

    clf, extractor, sim = build_pipeline(bag_of_words)
    data, labels = extractor.generate_features(records)
    data = vec.fit_transform(data).toarray()

    # load similarities for use in density sampling
    #sim = pickle.load(open('pickles/similarities_all.p', 'rb'))
    #sim = get_similarities(records, extractor)

    # set up splits to test on
    sets = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=splits, random_state=seed)

    r_accuracy, r_precision, r_recall, r_fscore = random_sampling(clf, data, labels, sets, splits, seed)
    u_accuracy, u_precision, u_recall, u_fscore = uncertainty_sampling(clf, data, labels, sets, splits, seed)
    d_accuracy, d_precision, d_recall, d_fscore = uncertainty_sampling(clf, data, labels, sets, splits, seed, sim)

    scores = [['Accuracy'], ['Precision'], ['Recall'], ['F-Score']]
    # accuracy scores
    scores[0].append(r_accuracy)
    scores[0].append(u_accuracy)
    scores[0].append(d_accuracy)
    # precision scores
    scores[1].append(r_precision)
    scores[1].append(u_precision)
    scores[1].append(d_precision)
    # recall scores
    scores[2].append(r_recall)
    scores[2].append(u_recall)
    scores[2].append(d_recall)
    # f scores
    scores[3].append(r_fscore)
    scores[3].append(u_fscore)
    scores[3].append(d_fscore)

    samples_per_split = len(data)/splits

    f_name = 'pickles/newCrossValidCurves_seed%s_splits%s.p' % (seed, splits)
    pickle.dump(scores, open(f_name, 'wb'))

    for i in xrange(4):
        draw_learning_comparison(splits, scores[i][1], scores[i][2], scores[i][3], samples_per_split, scores[i][0])


if __name__ == '__main__':
    '''
    learning_comparison(splits=5, seed=2, which_set='original', bag_of_words=True)
    learning_comparison(splits=10, seed=2, which_set='original', bag_of_words=True)
    learning_comparison(splits=20, seed=2, which_set='original', bag_of_words=True)
    #learning_comparison(40, which_set='original')
    #pickle_similarities(which_set='original')
    '''
    scores = pickle.load(open('new_plots/seed1_splits5.p', 'rb'))
    print scores

