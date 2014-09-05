import sqlite3
import random
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped


def learning_curves(repeats=10):
    """
    Plot learning curve thingies
    """
    data, labels = load_features_data(eu_adr_only=False)
    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(data).toarray()

    clf = build_pipeline('rbf', optimise_params=True, data=data, labels=labels)

    samples_per_split = len(data)/10
    scores = np.zeros(shape=(repeats, 9, 3, 2))
    accuracy = np.zeros(shape=(repeats, 9))

    for i in xrange(repeats):
        scores[i], accuracy[i] = get_data_points(clf, data, labels, i)

    # now need to average it out somehow
    av_scores = scores.mean(axis=0)
    av_accuracy = accuracy.mean(axis=0)
    draw_plots(av_scores, av_accuracy, samples_per_split)

    #pickle.dump(scores, open('scores.p', 'wb'))
    #pickle.dump(av_scores, open('av_scores.p', 'wb'))


def load_features_data(eu_adr_only=False):
    """
    Load some part of data
    """
    # set up feature extractor with desired features
    extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True)
    with sqlite3.connect('database/euadr_biotext.db') as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

    # may only want to look at sentences from eu-adr to start with
    if eu_adr_only:
        cursor.execute('''SELECT relations.*
                              FROM relations NATURAL JOIN sentences
                              WHERE sentences.source = 'eu-adr';''')
    else:
        # want to create features for all relations in db, training test split will be done by scikit-learn
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')

    records = cursor.fetchall()
    feature_vectors, class_vector = extractor.generate_features(records, balance_classes=False)

    return feature_vectors, class_vector


def build_pipeline(type, optimise_params=False, data=None, labels=None):
    """
    Set up classfier here to avoid repetition
    """
    # perform grid search on parameters if desired
    if type == 'linear':
        if optimise_params:
            optimal = tune_parameters(type, data, labels)
            best_c = optimal.named_steps['svm'].C

            # set up pipeline to normalise the data then build the model
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='linear', cache_size=1000, C=best_c))])
        # otherwise just use default values
        else:
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='linear', cache_size=1000, C=0.1))])
    elif type == 'poly':
        pass
    else:
        if optimise_params:
            optimal = tune_parameters(type, data, labels)
            best_c = optimal.named_steps['svm'].C
            best_gamma = optimal.named_steps['svm'].gamma

            # set up pipeline to normalise the data then build the model
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='rbf', cache_size=1000, C=best_c, gamma=best_gamma))])
        # otherwise just use default values
        else:
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='rbf', cache_size=1000, C=1, gamma=1))])

    return clf


def tune_parameters(svm_type, data, labels):
    """
    Tune the parameters using exhaustive grid search
    """
    # set cv here, why not
    cv = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True)

    if svm_type == 'linear':
        pipeline = Pipeline([('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='linear', cache_size=1000))])

        # only have error value to play with for linear kernel
        param_grid = [{'svm__C': np.linspace(0.1, 1, 10)}]
    elif svm_type == 'poly':
        pass
    else:
        pipeline = Pipeline([('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='rbf', cache_size=1000))])

        # only have error value to play with for linear kernel
        param_grid = [{'svm__C': np.linspace(0.1, 1, 10), 'svm__gamma': np.logspace(-6, -1, 10)}]

    print 'tuning params'
    clf = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv)
    clf.fit(data, labels)

    print 'best parameters found:'
    print clf.best_estimator_
    return clf.best_estimator_


def get_data_points(clf, data, labels, j):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    """
    # TODO fix this to use (stratified) k fold
    # set up array to hold scores
    scores = np.zeros(shape=(9, 3, 2))
    accuracy = np.zeros(shape=9)

    # first split at 10%
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, train_size=0.1,
                                                                                         random_state=j)
    #random_state=None)
    no_samples = len(train_data)
    scores[0], accuracy[0] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    # now loop to create remaining training sets
    for i in xrange(1, 9):
        more_data, test_data, more_labels, test_labels = cross_validation.train_test_split(test_data, test_labels,
                                                                                           train_size=no_samples,
                                                                                           random_state=i*j)
        #random_state=None)
        # add the new training data to existing
        train_data = np.append(train_data, more_data, axis=0)
        train_labels = np.append(train_labels, more_labels)
        scores[i], accuracy[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy


def get_scores(clf, train_data, train_labels, test_data, test_labels):
    """
    Return array of scores
    """
    # set up classifier and train
    clf.fit(train_data, train_labels)
    # calculate mean accuracy since not included in other set of scores
    accuracy = clf.score(test_data, test_labels)
    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    scores = precision_recall_fscore_support(test_labels, predicted)

    # return precision, recall and f1
    return np.array([scores[0], scores[1], scores[2]]), accuracy


def draw_plots(scores, av_accuracy, samples_per_split):
    """
    Create plots for precision, recall and f-score
    """
    #scores = pickle.load(open('av_scores.p', 'rb'))
    false_p = [s[0][0] for s in scores]
    true_p = [s[0][1] for s in scores]
    false_r = [s[1][0] for s in scores]
    true_r = [s[1][1] for s in scores]
    false_f = [s[2][0] for s in scores]
    true_f = [s[2][1] for s in scores]

    # create ticks for x axis
    ticks = np.linspace(samples_per_split, 9*samples_per_split, 9)

    plot(ticks, true_p, false_p, 'Precision', 'plots/' + time_stamped('balanced_precision.png'))
    plot(ticks, true_r, false_r, 'Recall', 'plots/' + time_stamped('balanced_recall.png'))
    plot(ticks, true_f, false_f, 'F-score', 'plots/' + time_stamped('balanced_fscore.png'))
    plot(ticks, av_accuracy, None, 'Accuracy', 'plots/' + time_stamped('balanced_accuracy.png'))


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

if __name__ == '__main__':
    learning_curves(repeats=20)
