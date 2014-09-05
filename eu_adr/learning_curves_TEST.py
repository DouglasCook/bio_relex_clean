import sqlite3
import operator
import pickle
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped

# TODO does this make sense? global thing here?
vec = DictVectorizer()
db_path = 'database/relex.db'


def load_records(eu_adr_only=False, orig_only=False):
    """
    Load some part of data
    """
    with sqlite3.connect(db_path) as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

    # may only want to look at sentences from eu-adr to start with
    if eu_adr_only:
        cursor.execute('''SELECT relations.*
                              FROM relations NATURAL JOIN sentences
                              WHERE sentences.source = 'eu-adr';''')
    elif orig_only:
        # want to create features for all eu-adr and biotext relations in db
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')
    else:
        # select all annotated relations
        cursor.execute('''SELECT relations.*
                          FROM relations
                          WHERE true_rel IS NOT NULL;''')

    records = cursor.fetchall()

    return records


def build_pipeline():
    """
    Set up classfier here to avoid repetition
    """
    clf = Pipeline([('normaliser', preprocessing.Normalizer(norm='l2')),
                    #('svm', SVC(kernel='rbf', gamma=10))])
                    #('svm', SVC(kernel='sigmoid'))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=2000, C=1000))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=3, gamma=2, cache_size=2000, C=1000))])
                    ('svm', SVC(kernel='rbf', gamma=10, cache_size=1000, C=1000))])
                    #('svm', SVC(kernel='linear'))])
                    #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False,
                    #n_jobs=-1))])
    return clf


def random_sampling(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, splits):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Folds are picked randomly
    """
    # set up array to hold scores
    scores = np.zeros(shape=(splits, 3, 2))
    accuracy = np.zeros(shape=splits)
    no_samples = len(train_indices)/splits

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    scores[0], accuracy[0] = get_scores(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices)

    # now loop to create remaining training sets
    for i in xrange(1, splits):
        # add the new training data to existing
        # NOW NEED TO TAKE TRAIN FIRST SINCE REST WILL CHANGE
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        scores[i], accuracy[i] = get_scores(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices)

    return scores, accuracy


def uncertainty_sampling(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, splits):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Samples the classifier is least confident about predicting are selected first
    """
    # set up array to hold scores
    scores = np.zeros(shape=(splits, 3, 2))
    accuracy = np.zeros(shape=splits)
    no_samples = len(train_indices)/splits

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    # want to copy it here so original data is not modified
    scores[0], accuracy[0], rest_data = get_scores(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, rest_indices)

    # now loop to create remaining training sets
    for i in xrange(1, splits):
        # calculate uncertainty of classifier on remaining data
        confidence = clf.decision_function(rest_data).flatten()
        # absolute value so both classes are considered
        confidence = np.absolute(confidence)

        # zip it all together and order by confidence
        remaining = sorted(zip(confidence, rest_indices), key=operator.itemgetter(0))
        #print 'remaining', len(remaining)
        confidence, rest_indices = zip(*remaining)

        # add the new training data to existing
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        scores[i], accuracy[i], rest_data = get_scores(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices,
                                                       rest_indices)

    return scores, accuracy


def density_sampling(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, sim, splits):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Samples selected based on confidence measure weighted by similarity to other samples
    """
    # set up array to hold scores
    scores = np.zeros(shape=(splits, 3, 2))
    accuracy = np.zeros(shape=splits)
    no_samples = len(train_indices)/splits

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    # want to copy it here so original data is not modified
    scores[0], accuracy[0], rest_data = get_scores(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, rest_indices)

    # now loop to create remaining training sets
    for i in xrange(1, splits):
        # calculate uncertainty of classifier on remaining data
        confidence = clf.decision_function(rest_data).flatten()
        # absolute value so both classes are considered
        confidence = np.absolute(confidence)

        # TODO may want to scale weighting between uncertainty and similarity score
        rest_sim = sim[np.array(rest_indices)]
        #rest_sim **= 0.8

        # weigh the confidence based on similarity measure
        confidence = np.multiply(confidence, rest_sim)
        # zip it all together and order by confidence
        remaining = sorted(zip(confidence, rest_indices), key=operator.itemgetter(0))
        #print 'remaining', len(remaining)
        confidence, rest_indices = zip(*remaining)

        # add the new training data to existing
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        scores[i], accuracy[i], rest_data = get_scores(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices,
                                                       rest_indices)

    return scores, accuracy


def get_similarities(vectors):
    """
    Calculate similarities of vectors ie one to all others
    """
    print 'calculating similarities'
    similarities = np.zeros(len(vectors))
    for i, v in enumerate(vectors):
        print i
        total = 0
        others = np.delete(vectors, i, 0)

        # loop through all other vectors and get total cosine distance
        for x in others:
            total += distance.cosine(v, x)

        # cos_similarity = 1 - av_cos_dist
        similarities[i] = 1 - total/len(others)
    print 'finished calculating similarities'

    return similarities


def pickle_similarities():
    """
    Pickle similarities based on all records
    """
    # TODO this is kind of wrong since the similarities will change as the word features are generated per split
    records = load_records()

    # set up extractor using desired features
    extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=5)
    extractor.create_dictionaries(records, how_many=5)

    data, _ = extractor.generate_features(records)
    data = vec.fit_transform(data).toarray()
    similarities = get_similarities(data)

    pickle.dump(similarities, open('pickles/similarities_all.p', 'wb'))


def get_scores(clf, extractor, records, orig_data, labels, train_indices, test_indices, rest_indices=None):
    """
    Return array of scores
    """
    # need to use list comprehension since records is not nice numpy array
    train_records = [records[i] for i in train_indices]

    # word features must be selected based on training set only otherwise test data contaminates training set
    extractor.create_dictionaries(train_records, how_many=5)

    # copy orig data here so it is not modified?
    data = orig_data

    # now add word features to the data
    #print data[0]
    #print len(data[0])
    extractor.generate_word_features(records, data)
    # convert from dict to array
    #print data[0]
    #print len(data[0])
    data = vec.fit_transform(data).toarray()

    train_data = data[train_indices]
    train_labels = labels[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    # for non random sampling need to return remaining data so confidence can be measured
    if rest_indices is not None:
        rest_data = data[np.array(rest_indices)]
    # TODO why does below result in wrong number of features?
    #train_data, train_labels = extractor.generate_features(train_records, balance_classes=False)
    #test_data, test_labels = extractor.generate_features(test_records, balance_classes=False)

    # train model
    clf.fit(train_data, train_labels)
    # calculate mean accuracy since not included in other set of scores
    accuracy = clf.score(test_data, test_labels)
    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    scores = precision_recall_fscore_support(test_labels, predicted)

    if rest_indices is not None:
        return np.array([scores[0], scores[1], scores[2]]), accuracy, rest_data
    else:
        return np.array([scores[0], scores[1], scores[2]]), accuracy


def learning_method_comparison(splits, repeats):
    """
    Plot learning curves to compare accuracy of different learning methods
    """
    clf = build_pipeline()
    # set up extractor using desired features
    extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=0)

    # want to have original records AND data
    records = load_records(eu_adr_only=False)
    orig_data, orig_labels = extractor.generate_features(records)

    # TODO what is the deal here???
    # this needs to match whatever percentage is being used for testing
    samples_per_split = (0.8/splits) * len(records)

    # if using density sampling only want to calculate similarities once
    sim = pickle.load(open('pickles/similarities_all.p', 'rb'))

    r_scores = np.zeros(shape=(repeats, splits, 3, 2))
    u_scores = np.zeros(shape=(repeats, splits, 3, 2))
    d_scores = np.zeros(shape=(repeats, splits, 3, 2))

    r_accuracy = np.zeros(shape=(repeats, splits))
    u_accuracy = np.zeros(shape=(repeats, splits))
    d_accuracy = np.zeros(shape=(repeats, splits))

    # run test with same starting conditions
    for i in xrange(repeats):
        print i
        # going to split the data here, then pass identical indices to the different learning methods
        all_indices = np.arange(len(records))

        # seed the shuffle here so can repeat experiment for different numbers of splits
        np.random.seed(2*i)
        np.random.shuffle(all_indices)

        # take off 20% for testing
        test_indices = all_indices[:len(records)/5]
        train_indices = all_indices[len(records)/5:]

        # split the data here using cross validator and return
        r_scores[i], r_accuracy[i] = random_sampling(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, splits)
        u_scores[i], u_accuracy[i] = uncertainty_sampling(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, splits)
        d_scores[i], d_accuracy[i] = density_sampling(clf, extractor, records, orig_data, orig_labels, train_indices, test_indices, sim,
                                                      splits)

    # create array of scores to pass to plotter
    scores = [['Accuracy'], ['Precision'], ['Recall'], ['F-Score']]
    # accuracy scores
    scores[0].append(r_accuracy.mean(axis=0, dtype=np.float64))
    scores[0].append(u_accuracy.mean(axis=0, dtype=np.float64))
    scores[0].append(d_accuracy.mean(axis=0, dtype=np.float64))

    # average over the repeats
    r_scores = r_scores.mean(axis=0, dtype=np.float64)
    u_scores = u_scores.mean(axis=0, dtype=np.float64)
    d_scores = d_scores.mean(axis=0, dtype=np.float64)
    # then true and false
    r_scores = r_scores.mean(axis=2, dtype=np.float64)
    u_scores = u_scores.mean(axis=2, dtype=np.float64)
    d_scores = d_scores.mean(axis=2, dtype=np.float64)

    # using numpy slicing to select correct scores
    for i in xrange(3):
        scores[i+1].append(r_scores[:, i])
        scores[i+1].append(u_scores[:, i])
        scores[i+1].append(d_scores[:, i])

    for i in xrange(4):
        draw_learning_comparison(splits, scores[i][1], scores[i][2], scores[i][3], samples_per_split, repeats,
                                 scores[i][0])


def draw_learning_comparison(splits, r_score, u_score, d_score, samples_per_split, repeats, scoring):
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
    plt.title('%s Comparison using %s batches and %s repeats' % (scoring, splits, repeats))

    plt.plot(ticks, r_score, label='Random Sampling')
    plt.plot(ticks, u_score, label='Uncertainty Sampling')
    plt.plot(ticks, d_score, label='Density Sampling')

    plt.legend(loc='best')

    plt.savefig('plots/learning_comparison_' + scoring + '_' + time_stamped('.png'), format='png')
    plt.clf()


if __name__ == '__main__':
    #pickle_similarities()
    start = time()
    learning_method_comparison(repeats=2, splits=5)
    end = time()
    print 'running time =', end - start
    #learning_method_comparison(repeats=20, splits=10)
    #learning_method_comparison(repeats=20, splits=20)
    #learning_method_comparison(repeats=20, splits=40)
