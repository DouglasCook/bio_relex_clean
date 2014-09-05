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
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped

db_path = 'database/relex.db'


def go(which):
    """
    Test the shiz
    """
    train, test = load_data()
    clf, extractor = build_pipeline(which, train)

    train_data, train_labels = extractor.generate_features(train, balance_classes=False)
    test_data, test_labels = extractor.generate_features(test, balance_classes=False)

    '''
    vec = DictVectorizer()
    # need to put together so features match
    all_data = np.append(train_data, test_data)
    all_data = vec.fit_transform(all_data).toarray()

    # slice into parts again
    train_data = all_data[:len(train_labels)]
    test_data = all_data[len(train_labels):]
    '''

    get_scores(clf, train_data, train_labels, test_data, test_labels)
    plot_roc_curve(clf, train_data, train_labels, test_data, test_labels)


def build_pipeline(which, train):
    """
    Set up classfier here to avoid repetition
    """
    if which == 'bag_of_words':
        clf = Pipeline([('vectoriser', DictVectorizer()),
                        #('scaler', preprocessing.StandardScaler(with_mean=False)),
                        ('normaliser', preprocessing.Normalizer(norm='l2')),
                        ('svm', LinearSVC(dual=True, C=1))])
        # set up extractor using desired features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, pos=False, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=True, bigrams=True)

    elif which == 'word_features':
        clf = Pipeline([('vectoriser', DictVectorizer(sparse=False)),
                        #('scaler', preprocessing.StandardScaler(with_mean=False)),
                        ('normaliser', preprocessing.Normalizer()),
                        #('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=10, C=1, cache_size=2000))])
                        #('svm', SVC(kernel='rbf', gamma=1, cache_size=1000, C=1))])
                        #('svm', SVC(kernel='linear', cache_size=1000, C=1))])
                        ('svm', LinearSVC(dual=True, C=1))])

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=True,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, bigrams=False)
        extractor.create_dictionaries(train, how_many=5)

    else:
        clf = Pipeline([('vectoriser', DictVectorizer(sparse=False)),
                        #('scaler', preprocessing.StandardScaler(with_mean=False)),
                        ('normaliser', preprocessing.Normalizer()),
                        #('svm', SVC(kernel='poly', coef0=1, degree=3, gamma=1, C=1, cache_size=2000))])
                        #('svm', SVC(kernel='rbf', gamma=100, cache_size=1000, C=10))])
                        #('svm', SVC(kernel='linear', cache_size=1000, C=1))])
                        ('svm', LinearSVC(dual=True, C=1))])

        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, word_features=False,
                                     combo=True, pos=True, entity_type=True, bag_of_words=False, bigrams=False)

    return clf, extractor


def load_data():
    """
    Load some part of data,
    """
    with sqlite3.connect(db_path) as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed';''')
        train = cursor.fetchall()

        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source = 'pubmed' AND
                                true_rel IS NOT NULL;''')
        test = cursor.fetchall()

    return train, test


def get_scores(clf, train_data, train_labels, test_data, test_labels):
    """
    Return array of scores
    """
    # set up classifier and train
    clf.fit(train_data, train_labels)

    # calculate mean accuracy since not included in other set of scores
    #accuracy = clf.score(test_data, test_labels)

    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate auroc and R, P, F scores
    #auroc = metrics.roc_auc_score(test_labels, predicted)
    #scores = precision_recall_fscore_support(test_labels, predicted)

    print metrics.classification_report(test_labels, predicted)
    print metrics.confusion_matrix(test_labels, predicted)

    #return np.array([scores[0], scores[1], scores[2]]), accuracy, auroc


def plot_roc_curve(clf, train_data, train_labels, test_data, test_labels):
    """
    Plot roc curve, not cross validated for now
    """
    clf.fit(train_data, train_labels)

    confidence = clf.decision_function(test_data)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, confidence)
    auroc = metrics.auc(fpr, tpr)

    # set up the figure
    plt.figure()
    #plt.grid()
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title('Receiver operating characteristic')
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], 'k--')

    plt.legend(loc='best')
    filepath = 'results/roc' + time_stamped('.png')
    plt.savefig(filepath, format='png')

if __name__ == '__main__':
    print 'BAG OF WORDS'
    go(which='bag_of_words')

    print 'WORD FEATURES'
    go(which='word_features')

    print 'FEATURES ONLY'
    go(which='poo')
