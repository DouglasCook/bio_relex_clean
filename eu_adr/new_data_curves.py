import sqlite3
import operator
import pickle
import math
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
from sklearn.svm import LinearSVC

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped

# TODO does this make sense? global thing here?
vec = DictVectorizer()
db_path = 'database/relex.db'


def load_records(orig_only):
    """
    Load original and new data sets
    """
    if orig_only:
        with sqlite3.connect('database/euadr_biotext_no_accents.db') as db:
            # using Row as row factory means can reference fields by name instead of index
            db.row_factory = sqlite3.Row
            cursor = db.cursor()

            # all records from original corpus
            cursor.execute('''SELECT relations.*
                              FROM relations NATURAL JOIN sentences
                              WHERE sentences.source != 'pubmed'
                              ORDER BY rel_id;''')

            orig = cursor.fetchall()
        return [], orig

    with sqlite3.connect(db_path) as db:
        # using Row as row factory means can reference fields by name instead of index
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # all records from original corpus
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source != 'pubmed'
                          ORDER BY rel_id;''')

        orig = cursor.fetchall()

        # now all newly annotated records
        cursor.execute('''SELECT relations.*
                          FROM relations NATURAL JOIN sentences
                          WHERE sentences.source = 'pubmed' AND
                                true_rel IS NOT NULL
                          ORDER BY rel_id;''')

        new = cursor.fetchall()

    return orig, new


def build_pipeline(bag_of_words, orig_only):
    """
    Set up classfier and extractor here to avoid repetition
    """
    if bag_of_words == 1:
        # BAG OF WORDS FEATURES
        clf = Pipeline([('vectoriser', DictVectorizer()),
                        ('normaliser', preprocessing.Normalizer(norm='l2')),
                        ('svm', LinearSVC(dual=True, C=1))])
        # set up extractor using desired features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, pos=False, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=True, bigrams=True)
        if orig_only:
            sim = pickle.load(open('pickles/orig_no_accents_similarities_bag_of_words.p', 'rb'))
        else:
            sim = pickle.load(open('pickles/similarities_bag_of_words.p', 'rb'))

    elif bag_of_words == 2:
        # ACTIVELY GENERATED WORD FEATURES
        clf = Pipeline([('vectoriser', DictVectorizer(sparse=False)),
                        ('normaliser', preprocessing.Normalizer(norm='l2')),
                        ('svm', LinearSVC(dual=True, C=1))])
        # set up extractor using desired features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, pos=True, combo=True,
                                     entity_type=True, word_features=True, bag_of_words=False, bigrams=False,
                                     after=False)
        if orig_only:
            sim = pickle.load(open('pickles/orig_no_accents_similarities_features_only.p', 'rb'))
        else:
            sim = pickle.load(open('pickles/similarities_features_only.p', 'rb'))

    elif bag_of_words == 3:
        # NON-WORD FEATURES RBF
        clf = Pipeline([('vectoriser', DictVectorizer(sparse=False)),
                        ('normaliser', preprocessing.Normalizer(norm='l2')),
                        ('svm', SVC(kernel='rbf', gamma=100, cache_size=2000, C=10))])
                        #('svm', SVC(kernel='poly', coef0=1, degree=3, gamma=2, cache_size=2000, C=1))])
                        #('svm', LinearSVC(dual=True, C=1))])
        # set up extractor using desired features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, pos=True, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=False, bigrams=False)
        if orig_only:
            sim = pickle.load(open('pickles/orig_no_accents_similarities_features_only.p', 'rb'))
        else:
            sim = pickle.load(open('pickles/similarities_features_only.p', 'rb'))

    else:
        # NON-WORD FEATURES
        clf = Pipeline([('vectoriser', DictVectorizer(sparse=False)),
                        ('normaliser', preprocessing.Normalizer(norm='l2')),
                        #('svm', SVC(kernel='rbf', gamma=100, cache_size=2000, C=10))])
                        ('svm', LinearSVC(dual=True, C=1))])
        # set up extractor using desired features
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, pos=True, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=False, bigrams=False)
        if orig_only:
            sim = pickle.load(open('pickles/orig_no_accents_similarities_features_only.p', 'rb'))
        else:
            sim = pickle.load(open('pickles/similarities_features_only.p', 'rb'))

    return clf, extractor, sim


def random_sampling(clf, extractor, orig_records, new_records, train_indices, test_indices, splits, word_features):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Folds are picked randomly
    """
    # set up array to hold scores
    #scores = np.zeros(shape=(splits, 3, 2))
    scores = np.zeros(shape=(splits, 3))
    accuracy = np.zeros(shape=splits)
    # floor division to guarantee correct number of splits
    no_samples = len(train_indices)/splits

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    scores[0], accuracy[0] = get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices,
                                        word_features=word_features)

    # now loop to create remaining training sets
    for i in xrange(1, splits - 1):
        # add the new training data to existing
        # NOW NEED TO TAKE TRAIN FIRST SINCE REST WILL CHANGE
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        #print 'random', len(train_indices)
        scores[i], accuracy[i] = get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices,
                                            word_features=word_features)

    # for last split add all remaining data
    train_indices = np.append(train_indices, rest_indices)
    scores[splits-1], accuracy[splits-1] = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                      test_indices, word_features=word_features)

    return scores, accuracy


def uncertainty_sampling(clf, extractor, orig_records, new_records, train_indices, test_indices, splits, word_features,
                         sim=None):
    """
    Get set of data points for one curve
    Want to add to the training data incrementally to mirror real life situtation
    Samples the classifier is least confident about predicting are selected first
    """
    # set up array to hold scores
    #scores = np.zeros(shape=(splits, 3, 2))
    scores = np.zeros(shape=(splits, 3))
    accuracy = np.zeros(shape=splits)
    # floor division to guarantee correct number of splits
    no_samples = len(train_indices)/splits
    print 'number samples', no_samples

    # now take first split for training and leave remainder
    # NEED TO TAKE REST FIRST SINCE TRAIN WILL CHANGE
    rest_indices = train_indices[no_samples:]
    train_indices = train_indices[:no_samples]

    scores[0], accuracy[0], rest_data = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                   test_indices, rest_indices, word_features)

    # now loop to create remaining training sets
    for i in xrange(1, splits - 1):
        # calculate uncertainty of classifier on remaining data
        confidence = clf.decision_function(rest_data).flatten()
        # absolute value so both classes are considered
        confidence = np.absolute(confidence)

        # if using density sampling
        if sim is not None:
            # load relevant similarities
            rest_sim = sim[np.array(rest_indices)]
            # TODO may want to scale weighting between uncertainty and similarity score
            # uses beta parameter as a power, linearly scaling will have no effect!
            #rest_sim **= 2

            # weigh the confidence based on similarity measure, lower confidence is preferred (since its distance)
            # divide by similarity since cosine similarity = 1 when vectors are the same
            #confidence = np.multiply(confidence, rest_sim)
            confidence = np.divide(confidence, rest_sim)

        # zip it all together and order by confidence
        remaining = sorted(zip(confidence, rest_indices), key=operator.itemgetter(0))
        confidence, rest_indices = zip(*remaining)

        # add the new training data to existing
        train_indices = np.append(train_indices, rest_indices[:no_samples])
        rest_indices = rest_indices[no_samples:]
        # calculate scores and return rest of data for use in sampling
        scores[i], accuracy[i], rest_data = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                       test_indices, rest_indices, word_features)

    print 'final batch', len(rest_indices)
    # last split done differently
    # add remaining data for last split so all data is used
    train_indices = np.append(train_indices, rest_indices)
    scores[splits-1], accuracy[splits-1] = get_scores(clf, extractor, orig_records, new_records, train_indices,
                                                      test_indices, None, word_features)

    return scores, accuracy


def get_similarities(records, extractor, orig_length):
    """
    Calculate similarities of vectors ie one to all others
    """
    data, _ = extractor.generate_features(records)
    data = vec.fit_transform(data).toarray()

    print 'calculating similarities'
    similarities = np.zeros(len(data))
    for i, v in enumerate(data):
        if i < orig_length:
            continue
        print i
        total = 0
        others = np.delete(data, i, 0)

        # loop through all other vectors and get total cosine distance
        for x in others:
            dist = distance.cosine(v, x)
            # check for NaN, not sure why this happens but ignore it
            if not math.isnan(dist):
                total += dist

        # cosine similarity = 1 - average cosine distance
        similarities[i] = 1 - total/len(others)

    return similarities


def pickle_similarities(orig_only, bag_of_words):
    """
    Pickle similarities for newly annotated data only
    """
    # TODO this is kind of wrong since the similarities will change as the word features are generated per split
    orig_records, new_records = load_records(orig_only)
    all_records = orig_records + new_records
    orig_length = len(orig_records)

    # set up extractor using desired features
    if bag_of_words:
        # FOR THE SPARSE LINEAR SVM
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, pos=False, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=True, bigrams=True)
        if orig_only:
            f_name = 'pickles/orig_no_accents_similarities_bag_of_words.p'
        else:
            f_name = 'pickles/similarities_bag_of_words.p'

    else:
        # FOR THE FEATURES ONE
        extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=True, pos=True, combo=True,
                                     entity_type=True, word_features=False, bag_of_words=False, bigrams=False)
        '''
        # BELOW FOR SPECIFIC WORD FEATURES
        #extractor.create_dictionaries(all_records, how_many=5)
        #extractor = FeatureExtractor(word_gap=False, count_dict=False, phrase_count=False, word_features=True)
        '''
        if orig_only:
            f_name = 'pickles/orig_no_accents_similarities_features_only.p'
        else:
            f_name = 'pickles/similarities_features_only.p'

    similarities = get_similarities(all_records, extractor, orig_length)

    # only want to pickle new data since orig always used for training
    pickle.dump(similarities[orig_length:], open(f_name, 'wb'))


def get_scores(clf, extractor, orig_records, new_records, train_indices, test_indices, rest_indices=None,
               word_features=0):
    """
    Return array of scores
    """
    # add sample of new records to original records (always used for training)
    train_records = [new_records[i] for i in train_indices] + orig_records
    test_records = [new_records[i] for i in test_indices]

    # if actively generated word features are being used
    if word_features:
        #word features must be selected based on training set only otherwise test data contaminates training set
        extractor.create_dictionaries(train_records, how_many=word_features)

    train_data, train_labels = extractor.generate_features(train_records)
    test_data, test_labels = extractor.generate_features(test_records)

    '''
    # need to smash everything together before generating features so same features are generated for each set
    if rest_indices is None:
        train_length = len(train_data)
        data = vec.fit_transform(train_data + test_data).toarray()
        train_data = data[:train_length]
        test_data = data[train_length:]
    else:
        rest_records = [new_records[i] for i in rest_indices]
        rest_data, _ = extractor.generate_features(rest_records)
        train_length = len(train_data)
        test_length = len(test_data)
        data = vec.fit_transform(train_data + test_data + rest_data).toarray()
        train_data = data[:train_length]
        test_data = data[train_length:train_length + test_length]
        rest_data = data[train_length + test_length:]
    '''

    if rest_indices is not None:
        rest_records = [new_records[i] for i in rest_indices]
        rest_data, _ = extractor.generate_features(rest_records)

    # train model
    clf.fit(train_data, train_labels)
    # calculate mean accuracy since not included in other set of scores
    accuracy = clf.score(test_data, test_labels)
    # classify the test data
    predicted = clf.predict(test_data)
    # evaluate accuracy of output compared to correct classification
    scores = precision_recall_fscore_support(test_labels, predicted, average='micro')

    # for non random sampling need to return remaining data so confidence can be measured
    if rest_indices is not None:
        return np.array([scores[0], scores[1], scores[2]]), accuracy, rest_data

    else:
        return np.array([scores[0], scores[1], scores[2]]), accuracy


def draw_learning_comparison(splits, r_score, u_score, d_score, samples_per_split, repeats, scoring, seed):
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

    f_name = 'plots/%s_new_learning_comparison_%s_%s' % (seed, scoring, time_stamped('.png'))
    plt.savefig(f_name, format='png')
    plt.clf()
    plt.close()


def learning_method_comparison(splits, repeats, seed, bag_of_words=0, orig_only=False, word_features=0):
    """
    Plot learning curves to compare accuracy of different learning methods
    """
    clf, extractor, sim = build_pipeline(bag_of_words, orig_only)

    # orig will always be use for training, new will be used for testing and added incrementally
    orig_records, new_records = load_records(orig_only)

    # samples per split = number of records remaining after removing test set divided by number of splits
    samples_per_split = (4*len(new_records)) / (5*splits)
    print 'samples per split', samples_per_split

    # TODO below used if similarities are to be generated each run (so extractor is def correct)
    #all_records = orig_records + new_records
    #sim = get_similarities(all_records, extractor)
    #sim = sim[len(orig_records):]

    #r_scores = np.zeros(shape=(repeats, splits, 3, 2))
    #u_scores = np.zeros(shape=(repeats, splits, 3, 2))
    #d_scores = np.zeros(shape=(repeats, splits, 3, 2))
    r_scores = np.zeros(shape=(repeats, splits, 3))
    u_scores = np.zeros(shape=(repeats, splits, 3))
    d_scores = np.zeros(shape=(repeats, splits, 3))

    r_accuracy = np.zeros(shape=(repeats, splits))
    u_accuracy = np.zeros(shape=(repeats, splits))
    d_accuracy = np.zeros(shape=(repeats, splits))

    # loop number of times to generate average scores
    for i in xrange(repeats):
        print i
        # going to split the data here, then pass identical indices to the different learning methods
        all_indices = np.arange(len(new_records))

        # seed the shuffle here so can repeat experiment for different numbers of splits
        np.random.seed(seed * i)
        np.random.shuffle(all_indices)

        # take off 20% for testing
        test_indices = all_indices[:len(new_records)/5]
        train_indices = all_indices[len(new_records)/5:]

        # now use same test and train indices to generate scores for each learning method
        u_scores[i], u_accuracy[i] = uncertainty_sampling(clf, extractor, orig_records, new_records, train_indices,
                                                          test_indices, splits, word_features)
        d_scores[i], d_accuracy[i] = uncertainty_sampling(clf, extractor, orig_records, new_records, train_indices,
                                                          test_indices, splits, word_features, sim)
        r_scores[i], r_accuracy[i] = random_sampling(clf, extractor, orig_records, new_records, train_indices,
                                                     test_indices, splits, word_features)

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
    #r_scores = r_scores.mean(axis=2, dtype=np.float64)
    #u_scores = u_scores.mean(axis=2, dtype=np.float64)
    #d_scores = d_scores.mean(axis=2, dtype=np.float64)

    # using numpy slicing to select correct scores
    for i in xrange(3):
        scores[i+1].append(r_scores[:, i])
        scores[i+1].append(u_scores[:, i])
        scores[i+1].append(d_scores[:, i])

    f_name = 'pickles/newCurves_seed%s_splits%s_' % (seed, splits)
    f_name = f_name + time_stamped('.p')
    pickle.dump(scores, open(f_name, 'wb'))

    for i in xrange(4):
        draw_learning_comparison(splits, scores[i][1], scores[i][2], scores[i][3], samples_per_split, repeats,
                                 scores[i][0], seed)


if __name__ == '__main__':
    start = time()
    #learning_method_comparison(repeats=10, splits=5)
    # CANNOT USE SEED ZERO
    learning_method_comparison(repeats=5, splits=5, seed=3, bag_of_words=0, orig_only=False, word_features=0)
    learning_method_comparison(repeats=5, splits=10, seed=3, bag_of_words=0, orig_only=False, word_features=0)
    learning_method_comparison(repeats=5, splits=20, seed=3, bag_of_words=0, orig_only=False, word_features=0)
    #learning_method_comparison(repeats=10, splits=5, seed=2, bag_of_words=False)
    #learning_method_comparison(repeats=10, splits=10, seed=2, bag_of_words=False)
    #learning_method_comparison(repeats=10, splits=20, seed=2, bag_of_words=False)
    #learning_method_comparison(repeats=20, splits=40)
    end = time()
    print 'running time =', end - start

    '''
    sim = pickle.load(open('pickles/similarities_bag_of_words.p', 'rb'))
    sim3 = pickle.load(open('pickles/similarities_features_only.p', 'rb'))
    pickle_similarities(orig_only=False, bag_of_words=False)
    pickle_similarities(orig_only=False, bag_of_words=True)

    pickle_similarities(orig_only=True, bag_of_words=False)
    pickle_similarities(orig_only=True, bag_of_words=True)
    '''
