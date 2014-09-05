import pickle
import datetime
import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV


def load_data(eu_adr_only=False, total_instances=0):
    """
    Load some part of data
    Biotext instances are at end of data set so will be sliced off and balance set
    """
    # TODO slice off a random part of the biotext examples? could use shuffle
    # use all instances if zero is passed in
    if total_instances == 0:
        # if eu_adr is requested do not slice anything off
        if eu_adr_only:
            features = pickle.load(open('pickles/scikit_data_eu_adr_only.p', 'rb'))
            labels = np.array(pickle.load(open('pickles/scikit_target_eu_adr_only.p', 'rb')))
        else:
            features = pickle.load(open('pickles/scikit_data.p', 'rb'))
            labels = np.array(pickle.load(open('pickles/scikit_target.p', 'rb')))
    # otherwise slice number of instances requested, biotext ones are at end so will be cut off
    else:
        features = pickle.load(open('pickles/scikit_data.p', 'rb'))

        # may want to load random selection of the biotext samples
        #biotext = features[751:]
        #random.shuffle(biotext)
        #features = features[:751] + biotext[:total_instances - 751]
        features = features[:total_instances]

        labels = pickle.load(open('pickles/scikit_target.p', 'rb'))
        labels = np.array(labels[:total_instances])

    return features, labels


def check_data_range(data):
    """
    Get some idea of how the data is distributed
    """
    print np.max(data)
    print np.min(data)
    print np.mean(data)


def cross_validated(eu_adr_only=False, total_instances=0):
    """
    Calculate stats using cross validations
    """
    features, labels = load_data(eu_adr_only=eu_adr_only, total_instances=total_instances)
    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # set up pipeline to normalise the data then build the model
    # TODO do I want normalise all of the features?
    clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                    # don't think scaling is required if I normalise the data?
                    #('scaler', preprocessing.MinMaxScaler()),
                    #('scaler', preprocessing.StandardScaler()),
                    #('svm', SVC(kernel='linear', C=2.5))])
                    #('svm', SVC(kernel='rbf', gamma=1))])
                    #('svm', SVC(kernel='sigmoid', gamma=10, coef0=10))])
                    ('svm', SVC(kernel='poly', coef0=3, degree=2, gamma=1, cache_size=1000))])
    print clf.get_params()['svm']

    # TODO what is the first parameter here?
    # when using cross validation there is no need to manually train the model
    cv = cross_validation.StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=0)

    with open('results/results.txt', 'a') as log:
        # TODO must be a better way to calculate the scores, ie not one at a time and separated by class???
        log.write('started: ' + str(datetime.datetime.now()) + '\n')

        # n_jobs parameter is number of cores to use, -1 for all cores
        log.write('accuracy = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='accuracy', n_jobs=-1)))
        log.write('precision = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='precision', n_jobs=-1)))
        log.write('recall = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='recall', n_jobs=-1)))
        log.write('F-score = %f\n' %
                  np.mean(cross_validation.cross_val_score(clf, data, labels, cv=cv, scoring='f1', n_jobs=-1)))

        log.write('finished: ' + str(datetime.datetime.now()) + '\n\n')


def no_cross_validation(eu_adr_only=False, total_instances=0, train_size=0.9):
    features, labels = load_data(eu_adr_only=eu_adr_only, total_instances=total_instances)

    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()
    # split data into training and test sets
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels,
                                                                                         train_size=train_size)

    # tune the parameters
    best_estimator = tune_parameters(train_data, train_labels)
    best_coef = best_estimator.named_steps['svm'].coef0
    best_degree = best_estimator.named_steps['svm'].degree
    best_gamma = best_estimator.named_steps['svm'].gamma

    # set up pipeline to normalise the data then build the model
    clf = Pipeline([('scaler', preprocessing.Normalizer()),
                    #('scaler', preprocessing.StandardScaler()),
                    #('svm', SVC(kernel='linear'))])
                    #('svm', SVC(kernel='rbf', gamma=1))])
                    #('svm', SVC(kernel='sigmoid', gamma=10, coef0=10))])
                    #('svm', SVC(kernel='poly', coef0=4, gamma=0.5, degree=2))])
                    ('svm', SVC(kernel='poly', coef0=best_coef, degree=best_degree, gamma=best_gamma,
                                cache_size=1000))])
    clf.fit(train_data, train_labels)

    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    print precision_recall_fscore_support(test_labels, predicted)
    print metrics.classification_report(test_labels, predicted)
    print metrics.confusion_matrix(test_labels, predicted)


def learning_curves(filepath, scoring, eu_adr_only=False, total_instances=0):
    """
    Plot learning curves of f-score for training and test data
    """
    features, labels = load_data(eu_adr_only, total_instances)

    # convert from dict into np array
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # set up pipeline to normalise the data then build the model
    clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                    ('svm', SVC(kernel='poly', coef0=3, degree=2, gamma=1, cache_size=1000))])
                    #('svm', SVC(kernel='linear'))])

    cv = cross_validation.StratifiedKFold(labels, n_folds=10, shuffle=True, random_state=0)

    # why does this always return results in the same pattern??? something fishy is going on
    # think that including 0.9 ends up in downward slope at the end
    sizes, t_scores, v_scores = learning_curve(clf, data, labels,
                                               train_sizes=np.linspace(0.1, 0.9, 8), cv=cv, scoring=scoring, n_jobs=-1)

    train_results = np.array([np.mean(t_scores[i]) for i in range(len(t_scores))])
    valid_results = np.array([np.mean(v_scores[i]) for i in range(len(v_scores))])
    '''
    # define new set of points to be used to smooth the plots
    x_new = np.linspace(sizes.min(), sizes.max())
    training_smooth = spline(sizes, training_results, x_new)
    validation_smooth = spline(sizes, validation_results, x_new)
    #plt.plot(sizes, validation_results)
    plt.plot(x_new, validation_smooth)
    #plt.plot(sizes, training_results)
    plt.plot(x_new, training_smooth)
    '''
    # instead lets fit a polynomial of degree ? as this should give a better impression!
    valid_coefs = np.polyfit(sizes, valid_results, deg=2)
    train_coefs = np.polyfit(sizes, train_results, deg=2)
    x_new = np.linspace(sizes.min(), sizes.max())
    valid_new = np.polyval(valid_coefs, x_new)
    train_new = np.polyval(train_coefs, x_new)

    # plot the raw points and the fitted curves
    #plt.plot(x_new, train_new)
    #plt.plot(sizes, train_results)
    plt.plot(x_new, valid_new, label='fitted poly degree 2')
    plt.plot(sizes, valid_results, label='raw points')

    kernel = str(clf.named_steps['svm'].get_params()['kernel'])
    coef = str(clf.named_steps['svm'].get_params()['coef0'])
    degree = str(clf.named_steps['svm'].get_params()['degree'])
    c_error = str(clf.named_steps['svm'].get_params()['C'])
    plt.title('kernel: ' + kernel + ', degree = ' + degree + ', coef = ' + coef + ', C = ' + c_error)
    plt.xlabel('training_instances')
    plt.ylabel('f_score')

    #plt.show()
    plt.savefig(filepath, format='tif')
    plt.clf()


def tune_parameters(data, labels):
    """
    Tune the parameters using exhaustive grid search
    """
    # set cv here, why not
    cv = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True)

    pipeline = Pipeline([('normaliser', preprocessing.Normalizer()),
                         ('svm', SVC(kernel='poly', gamma=1, cache_size=1000))])

    # can test multiple kernels as well if desired
    #param_grid = [{'kernel': 'poly', 'coef0': [1, 5, 10, 20], 'degree': [2, 3, 4, 5, 10]}]
    param_grid = [{'svm__coef0': [1, 2, 3, 4, 5], 'svm__degree': [2, 3, 4, 5]}]
    clf = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv)
    clf.fit(data, labels)

    print 'best parameters found:'
    print clf.best_estimator_
    return clf.best_estimator_


def make_curves():
    '''
    learning_curves('plots/eu_adr_only_f1.tif', 'f1', eu_adr_only=True)
    print 'bam'
    learning_curves('plots/eu_adr_only_accuracy.tif', 'accuracy', eu_adr_only=True)
    print 'bam'
    learning_curves('plots/eu_adr_only_precision.tif', 'precision', eu_adr_only=True)
    print 'bam'
    learning_curves('plots/eu_adr_only_recall.tif', 'recall', eu_adr_only=True)
    print 'bam'
    learning_curves('plots/all_f1.tif', 'f1', eu_adr_only=False)
    print 'bam'
    learning_curves('plots/all_accuracy.tif', 'accuracy', eu_adr_only=False)
    print 'bam'
    learning_curves('plots/all_precision.tif', 'precision', eu_adr_only=False)
    print 'bam'
    learning_curves('plots/all_recall.tif', 'recall', eu_adr_only=False)
    print 'bam'
    '''
    learning_curves('plots/balanced_f1.tif', 'f1', eu_adr_only=False, total_instances=1150)
    print 'bam'
    learning_curves('plots/balanced_accuracy.tif', 'accuracy', eu_adr_only=False, total_instances=1150)
    print 'bam'
    learning_curves('plots/balanced_precision.tif', 'precision', eu_adr_only=False, total_instances=1150)
    print 'bam'
    learning_curves('plots/balanced_recall.tif', 'recall', eu_adr_only=False, total_instances=1150)
    print 'bam'


def compare_datasets():
    cross_validated(eu_adr_only=True, total_instances=0)
    cross_validated(eu_adr_only=False, total_instances=0)
    cross_validated(eu_adr_only=False, total_instances=1150)


if __name__ == '__main__':
    #cross_validated(1150)
    #learning_curves(total_instances=1150)
    #learning_curves(eu_adr_only=True)
    #make_curves()
    #compare_datasets()
    no_cross_validation(eu_adr_only=False, total_instances=0, train_size=0.8)

