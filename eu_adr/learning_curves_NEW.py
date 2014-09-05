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


def learning_curves(splits, repeats):
    """
    Plot learning curve thingies
    """
    clf = build_pipeline()
    data, labels = load_features_data(eu_adr_only=False)

    samples_per_split = 0.09*len(data)
    scores = np.zeros(shape=(repeats, 10, 3, 2))
    accuracy = np.zeros(shape=(repeats, 10))

    for i in xrange(repeats):
        #scores[i], accuracy[i] = get_data_points(clf, data, labels, splits, i)
        scores[i], accuracy[i] = uncertainty_sampling(clf, data, labels, splits, i)
        #scores[i], accuracy[i] = random_sampling(clf, data, labels, splits, i)

    # now need to average it out somehow
    av_scores = scores.mean(axis=0, dtype=np.float64)
    av_accuracy = accuracy.mean(axis=0, dtype=np.float64)
    draw_true_false_plots(av_scores, av_accuracy, samples_per_split)

    #pickle.dump(scores, open('scores.p', 'wb'))
    #pickle.dump(av_scores, open('av_scores.p', 'wb'))


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
                    #('svm', SVC(kernel='poly', coef0=1, degree=3, gamma=2, cache_size=2000, C=10000))])
                    ('svm', SVC(kernel='rbf', gamma=30, cache_size=1000, C=1000))])
                    #('svm', SVC(kernel='linear'))])
                    #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False,
                    #n_jobs=-1))])
    return clf


def random_sampling(clf, extractor, records, train_indices, test_indices, splits):
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

    scores[0], accuracy[0] = get_scores(clf, extractor, records, train_indices, test_indices)

    # now loop to create remaining training sets
    for i in xrange(1, splits):
        # add the new training data to existing
        # NOW NEED TO TAKE TRAIN FIRST SINCE REST WILL CHANGE
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        #print 'random', len(train_indices)
        scores[i], accuracy[i] = get_scores(clf, extractor, records, train_indices, test_indices)

    return scores, accuracy


def uncertainty_sampling(clf, extractor, records, train_indices, test_indices, splits):
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

    scores[0], accuracy[0], rest_data = get_scores(clf, extractor, records, train_indices, test_indices, rest_indices)

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
        #print 'uncertainty', len(train_indices)

        if len(rest_indices) > 0:
            scores[i], accuracy[i], rest_data = get_scores(clf, extractor, records, train_indices, test_indices,
                                                           rest_indices)
        else:
            scores[i], accuracy[i] = get_scores(clf, extractor, records, train_indices, test_indices, None)

    return scores, accuracy


def density_sampling(clf, extractor, records, train_indices, test_indices, sim, splits):
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

    scores[0], accuracy[0], rest_data = get_scores(clf, extractor, records, train_indices, test_indices, rest_indices)

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
        #print 'density', len(train_indices)

        if len(rest_indices) > 0:
            scores[i], accuracy[i], rest_data = get_scores(clf, extractor, records, train_indices, test_indices,
                                                           rest_indices)
        else:
            scores[i], accuracy[i] = get_scores(clf, extractor, records, train_indices, test_indices, None)

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


def get_data_points(clf, data, labels, j):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Fold are picked randomly
    """
    # set up array to hold scores
    scores = np.zeros(shape=(9, 3, 2))
    accuracy = np.zeros(shape=9)

    # first split at 10%
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, train_size=0.1,
                                                                                         #random_state=j)
                                                                                         random_state=None)
    no_samples = len(train_data)
    scores[0], accuracy[0] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    # now loop to create remaining training sets
    for i in xrange(1, 9):
        more_data, test_data, more_labels, test_labels = cross_validation.train_test_split(test_data, test_labels,
                                                                                           train_size=no_samples,
                                                                                           #random_state=i*j)
                                                                                           random_state=None)
        # add the new training data to existing
        train_data = np.append(train_data, more_data, axis=0)
        train_labels = np.append(train_labels, more_labels)
        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy


def nicer_get_data_points(clf, data, labels, j):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    """
    # TODO problem here is with the random state, how can I make the experiment repeatable?
    # set up arrays to hold scores and indices
    scores = np.zeros(shape=(9, 3, 2))
    accuracy = np.zeros(shape=9)
    train_indices = np.array([], dtype=int)

    # set up stratified 10 fold cross validator, use specific random state for proper comparison
    # passing specific random state in here means same split is used every time
    cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=None)

    # iterating through the cv gives lists of indices for each fold
    # use test set for training since it is 10% of total data
    # TODO better way to do this?
    for i, (_, train) in enumerate(cv):
        if i == 9:
            print train_indices
            break
        # first add new fold to existing for training data and labels
        train_indices = np.append(train_indices, train)
        train_data = data[train_indices]
        train_labels = labels[train_indices]

        # then use complement for testing
        test_data = np.delete(data, train_indices, 0)
        test_labels = np.delete(labels, train_indices, 0)

        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy


def get_scores(clf, extractor, records, train_indices, test_indices, rest_indices=None):
    """
    Return array of scores
    """
    # need to use list comprehension since records is not nice numpy array
    train_records = [records[i] for i in train_indices]

    # word features must be selected based on training set only otherwise test data contaminates training set
    extractor.create_dictionaries(train_records, how_many=5)

    data, labels = extractor.generate_features(records)
    #print len(data[0])
    # convert from dict to array
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


def draw_true_false_plots(scores, av_accuracy, samples_per_split):
    """
    Create plots for precision, recall and f-score
    """
    #scores = pickle.load(open('av_scores.p', 'rb'))
    # TODO change this to use numpy slices eg [:, 0]
    false_p = [s[0][0] for s in scores]
    true_p = [s[0][1] for s in scores]
    false_r = [s[1][0] for s in scores]
    true_r = [s[1][1] for s in scores]
    false_f = [s[2][0] for s in scores]
    true_f = [s[2][1] for s in scores]

    # create ticks for x axis
    ticks = np.linspace(samples_per_split, 10*samples_per_split, 10)

    '''
    plot(ticks, true_p, false_p, 'Precision', 'plots/' + time_stamped('precision_2j_random.png'))
    plot(ticks, true_r, false_r, 'Recall', 'plots/' + time_stamped('recall_2j_random.png'))
    plot(ticks, true_f, false_f, 'F-score', 'plots/' + time_stamped('fscore_2j_random.png'))
    plot(ticks, av_accuracy, None, 'Accuracy', 'plots/' + time_stamped('accuracy_2j_random.png'))
    '''

    plot(ticks, true_p, false_p, 'Precision', 'plots/uncertainty_comparison/precision_3j_random.png')
    plot(ticks, true_r, false_r, 'Recall', 'plots/uncertainty_comparison/recall_3j_random.png')
    plot(ticks, true_f, false_f, 'F-score', 'plots/uncertainty_comparison/fscore_3j_random.png')
    plot(ticks, av_accuracy, None, 'Accuracy', 'plots/uncertainty_comparison/accuracy_3j_random.png')


def plot(ticks, true, false, scoring, filepath):
    """
    Plot given values
    """
    # set up the figure
    plt.figure()
    plt.grid()
    plt.xlabel('Training Instances')
    plt.ylabel('Score')
    plt.title(scoring)

    # if false not none then we are dealing with normal scores
    if false:
        # plot raw data points
        plt.plot(ticks, true, label='True relations')
        plt.plot(ticks, false, label='False relations')
    # else must be accuracy
    else:
        plt.plot(ticks, true, label='Average accuracy')

    # now fit polynomial (straight line) to the points and extend plot out
    x_new = np.linspace(ticks.min(), 2*ticks.max())
    true_coefs = np.polyfit(ticks, true, deg=1)
    true_fitted = np.polyval(true_coefs, x_new)
    plt.plot(x_new, true_fitted)

    # only plot false if not on accuracy score
    if false:
        false_coefs = np.polyfit(ticks, false, deg=1)
        false_fitted = np.polyval(false_coefs, x_new)
        plt.plot(x_new, false_fitted)

    plt.legend(loc='best')

    plt.savefig(filepath, format='png')
    plt.clf()


def learning_method_comparison(splits, repeats):
    """
    Plot learning curves to compare accuracy of different learning methods
    """
    clf = build_pipeline()
    # set up extractor using desired features
    extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=5)

    # want to have original records AND data
    records = load_records(eu_adr_only=False, orig_only=False)
    #print 'total records', len(records)

    # TODO what is the deal here???
    # this needs to match whatever percentage is being used for testing
    #samples_per_split = (0.8/splits) * len(records)
    samples_per_split = 4 * len(records)/(5 * splits)
    #print 'samples per split', samples_per_split

    # if using density sampling only want to calculate similarities once
    sim = pickle.load(open('pickles/similarities_all.p', 'rb'))

    r_scores = np.zeros(shape=(repeats, splits, 3, 2))
    u_scores = np.zeros(shape=(repeats, splits, 3, 2))
    d_scores = np.zeros(shape=(repeats, splits, 3, 2))

    r_accuracy = np.zeros(shape=(repeats, splits))
    u_accuracy = np.zeros(shape=(repeats, splits))
    d_accuracy = np.zeros(shape=(repeats, splits))

    # loop number of times to generate average scores
    for i in xrange(repeats):
        print i
        # going to split the data here, then pass identical indices to the different learning methods
        all_indices = np.arange(len(records))

        # seed the shuffle here so can repeat experiment for different numbers of splits
        np.random.seed(5*i)
        np.random.shuffle(all_indices)

        # take off 20% for testing
        # want the training set to be of a nice fixed size, rounding errors go into test set
        # this means that learning curves will always start and finish at same points
        test_indices = all_indices[4*(len(records)/5):]
        #print 'testing', len(test_indices)
        train_indices = all_indices[:4*(len(records)/5)]
        #print 'training', len(train_indices)

        # now use same test and train indices to generate scores for each learning method
        u_scores[i], u_accuracy[i] = uncertainty_sampling(clf, extractor, records, train_indices, test_indices, splits)
        r_scores[i], r_accuracy[i] = random_sampling(clf, extractor, records, train_indices, test_indices, splits)
        d_scores[i], d_accuracy[i] = density_sampling(clf, extractor, records, train_indices, test_indices, sim,
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
    learning_method_comparison(repeats=10, splits=5)
    #learning_method_comparison(repeats=20, splits=5)
    #learning_method_comparison(repeats=20, splits=10)
    #learning_method_comparison(repeats=20, splits=20)
    #learning_method_comparison(repeats=20, splits=40)
    end = time()
    print 'running time =', end - start

    #start = time()
    #learning_method_comparison(repeats=2, splits=5)
    #end = time()
    #print 'running time =', end - start

