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

if __name__ == '__main__':
    # used to run set of tests on VM
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
