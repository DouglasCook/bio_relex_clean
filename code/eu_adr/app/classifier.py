import sqlite3
import random

from sklearn.feature_extraction import DictVectorizer

import utility


class Classifier():
    """
    Classifier class to make pubmed classification more straight forward
    """
    # set up connection to database
    db_path = utility.build_filepath(__file__, '../database/relex.db')

    # set up vectoriser for transforming data from dictionary to numpy array
    vec = DictVectorizer()

    def __init__(self, f_extractor, use_db=False, optimise_params=False, no_biotext=False):
        # save feature extractor passed to constructor
        self.extractor = f_extractor

        # if want to save training set and predictions to database
        if use_db:
            # set up user record for the classifier
            self.user_id = self.create_user()

            # save records used for training for later experimentation with different classifiers
            self.create_training_set(no_biotext)

            # set up classifier pipeline, uses train function implemented in subclasses
            self.clf = self.train(optimise_params)

    def create_user(self):
        """
        Create new user for this classifier and return user id
        """
        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = db.cursor()
            cursor.execute('''INSERT INTO users
                                     VALUES (NULL, 'classifier', 'classifier');''')

            cursor.execute('SELECT MAX(user_id) as max FROM users;')

            return cursor.fetchone()['max']

    def create_training_set(self, no_biotext):
        """
        Write relations to use for training into classifier table
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # by default want to train on all examples with a 'correct' classification
            if no_biotext:
                print 'NO BIOTEXT!'
                cursor.execute('''INSERT INTO classifier_data
                                         SELECT ? as clsf_id,
                                                rel_id
                                         FROM relations NATURAL JOIN sentences
                                         WHERE true_rel IS NOT NULL AND
                                               sentences.source != 'biotext';''', [self.user_id])
            else:
                cursor.execute('''INSERT INTO classifier_data
                                         SELECT ? as clsf_id,
                                                rel_id
                                         FROM relations
                                         WHERE true_rel IS NOT NULL;''', [self.user_id])

    def get_training_data(self):
        """
        Return features and class labels for training set
        """
        # first balance the classes - NOT DOING THIS
        #self.balance_classes()
        # if want to balance need to change query below to use FROM classifier_data_balanced

        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            cursor = db.cursor()
            # write balanced sets of training rels to table
            cursor.execute('''SELECT *
                              FROM relations
                              WHERE rel_id IN (SELECT training_rel
                                               FROM classifier_data
                                               WHERE clsf_id = ?);''', [self.user_id])
            records = cursor.fetchall()

        # extract the feature vectors and class labels for training set
        return self.extractor.generate_features(records)

    def balance_classes(self):
        """
        Undersample the over-represented class so it contains same number of samples
        """
        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # first get everything that has been classified so far
            cursor.execute('''SELECT true_rel, rel_id
                              FROM relations
                              WHERE rel_id IN (SELECT training_rel
                                               FROM classifier_data
                                               WHERE clsf_id = ?);''', [self.user_id])
            records = cursor.fetchall()

        # count occurrence of each class
        true = [x for x in records if x[0] == 1]
        false = [x for x in records if x[0] == 0]
        true_count, false_count = len(true), len(false)

        # undersample the over represented class
        if true_count < false_count:
            false = random.sample(false, true_count)
        elif false_count < true_count:
            true = random.sample(true, false_count)

        # put back together again
        print len(true), len(false)
        together = [f[1] for f in false] + [t[1] for t in true]

        with sqlite3.connect(self.db_path) as db:
            cursor = db.cursor()
            # loop through all records and add to proper training table
            for t in together:
                cursor.execute('''INSERT INTO classifier_data_balanced
                                  VALUES (?, ?);''', (self.user_id, t))
