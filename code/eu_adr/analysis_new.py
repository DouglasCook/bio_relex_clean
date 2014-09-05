import sqlite3
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped


def create_results():
    """
    Get cross validated scores for classifiers built with various parameters
    Write results to csv file for easy human analysis
    """
    # test a variety of features and algorithms
    clf = build_pipeline()

    # set up output file
    with open('results/feature_selection/bigram_features.csv', 'wb') as f_out:
        csv_writer = csv.writer(f_out, delimiter=',')
        csv_writer.writerow(['features', 'accuracy', 'auroc', 'true_P', 'true_R', 'true_F',
                             'false_P', 'false_R', 'false_F', 'average_P', 'average_R', 'average_F'])

        # NEW SHIZ
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=False, pos=False, entity_type=False, bag_of_words=True, bigrams=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words + bigrams')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=False, pos=False, entity_type=True, bag_of_words=True, bigrams=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words + bigrams + type')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=True, bigrams=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words + bigrams + type + combo set')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=True, bigrams=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words + bigrams + type + combo_set + phrase')
        print 'done'

        '''
        # BAG OF WORDS
        # first using all words and no other features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=False, pos=False, entity_type=False, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=False, pos=False, entity_type=True, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words, type')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=False, word_features=False,
                                     combo=False, pos=False, entity_type=False, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words, gap')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=False, pos=False, entity_type=False, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words, phrase count')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=False, pos=True, entity_type=False, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words, non-count pos')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=True, pos=False, entity_type=False, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words, non-count combo')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=True, phrase_count=False, word_features=False,
                                     combo=False, pos=True, entity_type=False, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words, count pos')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=True, phrase_count=False, word_features=False,
                                     combo=True, pos=False, entity_type=False, bag_of_words=True)
        write_scores(csv_writer, clf, extractor, -1, 'bag of words, count combo')
        print 'done'
        '''

        '''
        # NO BAG OF WORDS
        # first using all words and no other features
        #extractor = FeatureExtractor(word_gap=True, count_dict=True, phrase_count=True, word_features=False,
                                     #combo=True, pos=True, entity_type=True, bag_of_words=False)
        #write_scores(csv_writer, clf, extractor, -1, 'all')
        #print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, -1, 'non-counting')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, -1, 'no gap')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=True, entity_type=False, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, -1, 'no type')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=False, word_features=False,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, -1, 'no phrase count')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, -1, 'no pos')
        print 'done'

        extractor = FeatureExtractor(word_gap=True, count_dict=False, phrase_count=True, word_features=False,
                                     combo=False, pos=True, entity_type=True, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, -1, 'no combo')
        print 'done'
        '''

        '''
        # SPECIFIC ACTIVELY SELECTED WORD FEATURES
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, 5, '5 words')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, before=False)
        write_scores(csv_writer, clf, extractor, 5, '5 words, no before')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, between=False)
        write_scores(csv_writer, clf, extractor, 5, '5 words, no between')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, after=False)
        write_scores(csv_writer, clf, extractor, 5, '5 words, no after')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False)
        write_scores(csv_writer, clf, extractor, 10, '10 words')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, before=False)
        write_scores(csv_writer, clf, extractor, 10, '10 words, no before')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, between=False)
        write_scores(csv_writer, clf, extractor, 10, '10 words, no between')
        print 'done'

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, after=False)
        write_scores(csv_writer, clf, extractor, 10, '10 words, no after')
        print 'done'
        '''


def build_pipeline():
    """
    Set up classfier here to avoid repetition
    """
    clf = Pipeline([('vectoriser', DictVectorizer()),
                    ('normaliser', preprocessing.Normalizer()),
                    ('svm', LinearSVC(dual=True))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=1000))])
                    #('svm', SVC(kernel='poly', coef0=1, degree=3, gamma=2, cache_size=1000, C=1000))])
                    #('svm', SVC(kernel='rbf', gamma=1, cache_size=1000))])
                    #('svm', SVC(kernel='rbf', gamma=1, cache_size=2000, C=10))])
                    #('svm', SVC(kernel='linear', cache_size=2000))])
                    #('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt', bootstrap=False,
                    # n_jobs=-1))])
    return clf


def write_scores(csv_writer, clf, extractor, how_many, features):
    """
    Write one set of scores to csv
    """
    scores, accuracy, auroc = cross_validated_scores(clf, extractor, how_many)

    for i in xrange(10):
        row = [features, accuracy[i], auroc[i],
               scores[i, 0, 1], scores[i, 1, 1], scores[i, 2, 1],  # true relations
               scores[i, 0, 0], scores[i, 1, 0], scores[i, 2, 0],  # false relations
               (scores[i, 0, 1] + scores[i, 0, 0])/2, (scores[i, 1, 1] + scores[i, 1, 0])/2,  # averages
               (scores[i, 2, 1] + scores[i, 2, 0])/2]

        csv_writer.writerow(row)


def cross_validated_scores(clf, extractor, how_many):
    """
    Calculate scores using 10 fold cross validation
    """
    # set up array to hold scores
    scores = np.zeros(shape=(10, 3, 2))
    accuracy = np.zeros(shape=10)
    auroc = np.zeros(shape=10)

    records = load_records()
    data, labels = extractor.generate_features(records)

    # set up stratified 10 fold cross validator, use specific random state for proper comparison
    cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=0)

    # iterating through the cv gives lists of indices for each fold
    for i, (train, test) in enumerate(cv):
        '''
        # set up word features based on training set only
        train_records = [records[j] for j in train]
        print len(train_records)
        extractor.create_dictionaries(train_records, how_many)
        #print extractor.bet_verb_dict

        # generate features here if actively selected words are being used
        data, labels = extractor.generate_features(records)
        #vec = DictVectorizer()
        #data = vec.fit_transform(data).toarray()
        '''

        train_data, test_data = data[train], data[test]
        train_labels, test_labels = labels[train], labels[test]

        scores[i], accuracy[i], auroc[i] = get_scores(clf, train_data, train_labels, test_data, test_labels)

    return scores, accuracy, auroc


def load_features_data(extractor, how_many):
    """
    Load some part of data
    """
    with sqlite3.connect('database/euadr_biotext.db') as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')

    records = cursor.fetchall()

    extractor.create_dictionaries(records, how_many)

    return extractor.generate_features(records, balance_classes=False)


def load_records():
    """
    Load some part of data
    """
    with sqlite3.connect('database/euadr_biotext.db') as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')

    records = cursor.fetchall()
    return records


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
    # evaluate auroc and R, P, F scores
    auroc = metrics.roc_auc_score(test_labels, predicted)
    scores = precision_recall_fscore_support(test_labels, predicted)

    # ROC STUFF
    #confidence = clf.decision_function(test_data)
    #fpr, tpr, thresholds = metrics.roc_curve(test_labels, confidence)
    #print fpr
    #print tpr
    #print thresholds

    return np.array([scores[0], scores[1], scores[2]]), accuracy, auroc


def plot_roc_curve():
    """
    Plot roc curve, not cross validated for now
    """
    clf = build_pipeline()
    extractor = FeatureExtractor(word_gap=True, word_features=True, count_dict=True, phrase_count=True)

    features, labels = load_features_data(extractor)
    # transform from dict into array for training
    vec = DictVectorizer()
    data = vec.fit_transform(features).toarray()

    # split data into train and test, may want to use cross validation later
    train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(data, labels, train_size=0.9,
                                                                                         random_state=1)
    clf.fit(train_data, train_labels)

    confidence = clf.decision_function(test_data)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, confidence)
    auroc = metrics.auc(fpr, tpr)

    print len(fpr), len(tpr)
    # set up the figure
    plt.figure()
    #plt.grid()
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title('Receiver operating characteristic')
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.legend(loc='best')
    filepath = 'results/' + time_stamped('roc.png')
    plt.savefig(filepath, format='png')


def tune_parameters(type):
    """
    Find best parameters for given kernels (and features)
    """
    records = load_records()

    if type == 'bag_of_words':
        print 'bag of words'
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=True, bigrams=True)
        data, labels = extractor.generate_features(records)
        cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=0)

        # use linear svm for sparse bag of words feature vector
        pipeline = Pipeline([('vectoriser', DictVectorizer()),
                            ('normaliser', preprocessing.Normalizer()),
                            ('svm', LinearSVC(dual=True))])

        param_grid = [{'svm__C': np.array([0.001, 0.1, 1, 10, 100])}]

    elif type == 'linear':
        print 'linear'
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=False)
        data, labels = extractor.generate_features(records)
        cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=0)

        # use linear svm for sparse bag of words feature vector
        pipeline = Pipeline([('vectoriser', DictVectorizer()),
                             ('normaliser', preprocessing.Normalizer()),
                             ('svm', LinearSVC(dual=True))])

        param_grid = [{'svm__C': np.array([1, 10, 100, 1000, 10000])}]

    elif type == 'rbf':
        print 'rbf'
        # non baog of words features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=False)
        data, labels = extractor.generate_features(records)
        cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=0)

        # use linear svm for sparse bag of words feature vector
        pipeline = Pipeline([('vectoriser', DictVectorizer()),
                             ('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='rbf', cache_size=2000))])

        param_grid = [{'svm__C': np.array([1, 10, 100, 1000, 10000]), 'svm__gamma': np.array([0.01, 0.1, 1, 100, 1000])}]

    elif type == 'poly':
        print 'poly'
        # non baog of words features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=False)
        data, labels = extractor.generate_features(records)
        cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=0)

        # use linear svm for sparse bag of words feature vector
        pipeline = Pipeline([('vectoriser', DictVectorizer()),
                             ('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='poly', cache_size=2000))])

        # grid search here takes a looong time
        param_grid = [{'svm__C': np.array([1, 10]), 'svm__gamma': np.array([1, 10]),
                      'svm__degree': np.array([2, 3, 4, 5]), 'svm__coef0': np.array([1, 2, 3, 4])}]

    elif type == 'sigmoid':
        print 'sigmoid'
        # non baog of words features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=False, entity_type=True, bag_of_words=False)
        data, labels = extractor.generate_features(records)
        cv = cross_validation.StratifiedKFold(labels, shuffle=True, n_folds=10, random_state=0)

        # use linear svm for sparse bag of words feature vector
        pipeline = Pipeline([('vectoriser', DictVectorizer()),
                             ('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='sigmoid', cache_size=2000))])

        param_grid = [{'svm__C': np.array([1, 10]), 'svm__gamma': np.array([0.001, 0.1, 1, 10]),
                       'svm__coef0': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}]

    clf = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=True)
    clf.fit(data, labels)
    print clf.best_estimator_
    print clf.best_params_
    print clf.best_score_


if __name__ == '__main__':
    #create_results()
    tune_parameters('bag_of_words')
    #tune_parameters('sigmoid')
    #tune_parameters('linear')
    #tune_parameters('rbf')
    #tune_parameters('poly')
    #plot_roc_curve()
