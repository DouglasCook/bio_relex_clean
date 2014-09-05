import sqlite3  # should be able to access this from parent but not working for some reason
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

from classifier import Classifier


class SVM(Classifier):
    """
    Support vector machine classifier
    """
    def classify(self, record):
        """
        Classify given record and write prediction and confidence measure to table
        """
        # list is expected when generating features so put the record in a list
        data = self.extractor.generate_features([record], no_class=True)
        data = self.vec.transform(data).toarray()

        # predict returns an array so need to remove element
        prediction = self.clf.predict(data)[0]
        # calculate distance from separating hyperplane as measure of confidence
        # zero zero index since only one element is being classified
        confidence = abs(self.clf.decision_function(data)[0][0])

        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            cursor.execute('''INSERT INTO predictions
                                     VALUES (NULL, ?, ?, ?, ?);''',
                           (record['rel_id'], self.user_id, prediction, confidence))


class SVMlinear(SVM):
    """
    SVM with linear kernel
    """
    def train(self, optimise_params, records=None):
        """
        Train the model on training set saved in db or on the records passed in
        """
        # if the records have been passed in directly use them
        if records:
            data, labels = self.extractor.generate_features(records)
        # otherwise use the data from database
        else:
            data, labels = self.get_training_data()
        # convert from dict into np array
        data = self.vec.fit_transform(data).toarray()

        # perform grid search on parameters if desired
        if optimise_params:
            optimal = self.tune_parameters(data, labels)
            best_c = optimal.named_steps['svm'].C

            # set up pipeline to normalise the data then build the model
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='linear', cache_size=1000, C=best_c))])
        # otherwise just use default values
        else:
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='linear', cache_size=1000))])

        # train the model
        clf.fit(data, labels)

        return clf

    @staticmethod
    def tune_parameters(data, labels):
        """
        Tune the parameters using exhaustive grid search
        """
        # set cv here, why not
        cv = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True)

        pipeline = Pipeline([('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='linear', cache_size=1000))])

        # only have error value to play with for linear kernel
        param_grid = [{'svm__C': np.linspace(0.1, 1, 10)}]
        print 'tuning params'
        clf = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv)
        clf.fit(data, labels)

        print 'best parameters found:'
        print clf.best_estimator_
        return clf.best_estimator_


class SVMpoly(SVM):
    """
    SVM with polynomial kernel
    """
    def train(self, optimise_params, records=None):
        """
        Train the model on selected training set
        """
        # if the records have been passed in directly use them
        if records:
            data, labels = self.extractor.generate_features(records)
        # otherwise use the data from database
        else:
            data, labels = self.get_training_data()
        # convert from dict into np array
        data = self.vec.fit_transform(data).toarray()

        # perform grid search on parameters if desired
        if optimise_params:
            optimal = self.tune_parameters(data, labels)
            best_coef = optimal.named_steps['svm'].coef0
            best_degree = optimal.named_steps['svm'].degree
            best_c = optimal.named_steps['svm'].C
            best_gamma = optimal.named_steps['svm'].gamma

            # set up pipeline to normalise the data then build the model
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='poly', coef0=best_coef, degree=best_degree, C=best_c, gamma=best_gamma,
                                        cache_size=1000))])
        # otherwise just use decided parameters
        else:
            # set up pipeline to normalise the data then build the model
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='poly', coef0=1, degree=2, gamma=1, cache_size=1000))])
        # train the model
        clf.fit(data, labels)

        return clf

    @staticmethod
    def tune_parameters(data, labels):
        """
        Tune the parameters using exhaustive grid search
        """
        # set cv here, why not
        cv = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True)

        pipeline = Pipeline([('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='poly', cache_size=1000))])

        param_grid = [{'svm__coef0': [1, 2, 3, 4, 5], 'svm__degree': [2, 3, 4, 5], 'svm__C': [1, 2],
                       'svm__gamma': [0, 1]}]

        print 'tuning params'
        clf = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv)
        clf.fit(data, labels)

        print 'best parameters found:'
        print clf.best_estimator_
        return clf.best_estimator_


class SVMrbf(SVM):
    """
    SVM with radial basis function kernel
    """
    def train(self, optimise_params, records=None):
        """
        Train the model on selected training set
        """
        # if the records have been passed in directly use them
        if records:
            data, labels = self.extractor.generate_features(records)
        # otherwise use the data from database
        else:
            data, labels = self.get_training_data()
        # convert from dict into np array
        data = self.vec.fit_transform(data).toarray()

        # perform grid search to optimise parameters if desired
        if optimise_params:
            optimal = self.tune_parameters(data, labels)
            best_c = optimal.named_steps['svm'].C
            best_gamma = optimal.named_steps['svm'].gamma

            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='rbf', gamma=best_gamma, C=best_c, cache_size=1000))])
        # otherwise use decided parameters
        else:
            clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                            ('svm', SVC(kernel='rbf', gamma=1, cache_size=1000))])

        # train the model
        clf.fit(data, labels)

        return clf

    @staticmethod
    def tune_parameters(data, labels):
        """
        Tune the parameters using exhaustive grid search
        """
        # set cv here, why not
        cv = cross_validation.StratifiedKFold(labels, n_folds=5, shuffle=True)

        pipeline = Pipeline([('normaliser', preprocessing.Normalizer()),
                             ('svm', SVC(kernel='rbf', cache_size=1000))])

        param_grid = [{'svm__C': [1, 2], 'svm__gamma': [0, 1]}]
        print 'tuning params'
        clf = GridSearchCV(pipeline, param_grid, n_jobs=-1, cv=cv)
        clf.fit(data, labels)

        print 'best parameters found:'
        print clf.best_estimator_
        return clf.best_estimator_
