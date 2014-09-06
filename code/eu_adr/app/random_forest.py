import sqlite3      # should be able to access this from parent but not working for some reason
from classifier import Classifier

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Classifier):
    """
    Random forest classifier
    """

    def train(self, records=None):
        """
        Train the model on selected training set
        """
        if records:
            data, labels = self.extractor.generate_features(records)
        # otherwise use the data from database
        else:
            data, labels = self.get_training_data()

        data = self.vec.fit_transform(data).toarray()

        clf = Pipeline([('normaliser', preprocessing.Normalizer()),
                        ('random_forest', RandomForestClassifier(n_estimators=10, max_features='sqrt',
                                                                 bootstrap=False, n_jobs=-1))])
        # train the model
        clf.fit(data, labels)

        return clf

    def classify(self, record):
        """
        Classify given record and write prediction and confidence measure to table
        """
        # list is expected when generating features so put the record in a list
        data = self.extractor.generate_features([record], no_class=True)
        data = self.vec.transform(data).toarray()

        # predict returns an array so need to remove element
        prediction = self.clf.predict(data)[0]

        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            cursor.execute('''INSERT INTO predictions
                                     VALUES (NULL, ?, ?, ?, 0);''',
                           (record['rel_id'], self.user_id, prediction))
