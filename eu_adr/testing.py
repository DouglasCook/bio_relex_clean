import sqlite3
import operator
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

from sklearn.feature_extraction import DictVectorizer

from app.feature_extractor import FeatureExtractor
from app.utility import time_stamped

import new_data_curves
import final_results

"""
vec = DictVectorizer()
db_path = 'database/relex.db'

with sqlite3.connect(db_path) as db:
    # using Row as row factory means can reference fields by name instead of index
    db.row_factory = sqlite3.Row
    cursor = db.cursor()

    # may only want to look at sentences from eu-adr to start with
    cursor.execute('''SELECT relations.*
                              FROM relations NATURAL JOIN sentences
                              WHERE true_rel IS NOT NULL;''')

    records = cursor.fetchall()

# first generate the features using old method
extractor = FeatureExtractor(word_features=5)
extractor.create_dictionaries(records, how_many=5)
data_orig, label_orig = extractor.generate_features(records)

extractor = FeatureExtractor()
data_new, labels_new = extractor.generate_features(records)
extractor.create_dictionaries(records, how_many=5)
extractor.generate_word_features(records, data_new)

#print data_orig
#print data_new

if data_orig == data_new:
    print 'booyah'
else:
    print 'nein!'
"""

if __name__ == '__main__':
    print 'orig bag'
    new_data_curves.learning_method_comparison(splits=5, repeats=30, seed=1, bag_of_words=1, orig_only=True, word_features=0)
    new_data_curves.learning_method_comparison(splits=10, repeats=30, seed=1, bag_of_words=1, orig_only=True, word_features=0)
    new_data_curves.learning_method_comparison(splits=20, repeats=30, seed=1, bag_of_words=1, orig_only=True, word_features=0)
    new_data_curves.learning_method_comparison(splits=40, repeats=30, seed=1, bag_of_words=1, orig_only=True, word_features=0)

    print 'orig rbf'
    new_data_curves.learning_method_comparison(splits=5, repeats=30, seed=1, bag_of_words=3, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=10, repeats=30, seed=1, bag_of_words=3, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=20, repeats=30, seed=1, bag_of_words=3, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=40, repeats=30, seed=1, bag_of_words=3, orig_only=False, word_features=0)

    print 'orig linear'
    new_data_curves.learning_method_comparison(splits=5, repeats=30, seed=1, bag_of_words=0, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=10, repeats=30, seed=1, bag_of_words=0, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=20, repeats=30, seed=1, bag_of_words=0, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=40, repeats=30, seed=1, bag_of_words=0, orig_only=False, word_features=0)

    print 'new bag'
    new_data_curves.learning_method_comparison(splits=5, repeats=30, seed=3, bag_of_words=1, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=10, repeats=30, seed=3, bag_of_words=1, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=20, repeats=30, seed=3, bag_of_words=1, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=40, repeats=30, seed=3, bag_of_words=1, orig_only=False, word_features=0)

    print 'new features'
    new_data_curves.learning_method_comparison(splits=5, repeats=30, seed=3, bag_of_words=0, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=10, repeats=30, seed=3, bag_of_words=0, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=20, repeats=30, seed=3, bag_of_words=0, orig_only=False, word_features=0)
    new_data_curves.learning_method_comparison(splits=40, repeats=30, seed=3, bag_of_words=0, orig_only=False, word_features=0)
