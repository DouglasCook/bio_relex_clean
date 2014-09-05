import sqlite3

import utility
from classifier import Classifier
from svm import SVMlinear, SVMpoly, SVMrbf
from random_forest import RandomForest

from feature_extractor import FeatureExtractor

db_path = utility.build_filepath(__file__, '../database/relex.db')


def update_correct_classifications():
    """
    Apply the decisions of the annotator(s) to the relations table
    """
    # TODO if there will be multiple annotators this needs to be changed to implement majority decision
    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # updated all unclassified relations based on annotators decisions
        # this query is horrific but don't think sqlite offers nicer way with joins in update query
        # decision < 2 so only true or false classifications are considered
        cursor.execute('''UPDATE relations
                          SET true_rel = (SELECT decisions.decision
                                          FROM decisions NATURAL JOIN users
                                          WHERE decision < 2 AND
                                                users.type != 'classifier' AND
                                                decisions.rel_id = relations.rel_id)
                          WHERE relations.rel_id IN (SELECT decisions.rel_id
                                                     FROM decisions NATURAL JOIN users
                                                     WHERE decisions.decision < 2 AND
                                                           users.type != 'classifier');''')


def classify_remaining(optimise_params=False, no_biotext=False):
    """
    Call classifier to predict values of remaining unclassified instances
    """
    # set up feature extractor with desired parameters
    f_extractor = FeatureExtractor()
    # set up classifier with link to feature extractor
    #clf = Classifier(f_extractor, optimise_params, no_biotext)
    clf = SVMpoly(f_extractor, use_db=True, optimise_params=optimise_params, no_biotext=no_biotext)

    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # query for all unclassified instances
        cursor.execute('''SELECT *
                          FROM relations
                          WHERE true_rel IS NULL''')
        # need to fetch all since classifier will want to use db and cannot have it locked
        records = cursor.fetchall()

    for row in records:
        clf.classify(row)


def count_true_false_predictions():
    """
    See how many relations predicted as true/false by latest run
    """
    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # get latest classifier
        cursor.execute('''SELECT max(user_id)
                          FROM users
                          WHERE type = 'classifier';''')
        clsf_id = cursor.fetchone()[0]

        # count the relations
        cursor.execute('''SELECT decision, count(rel_id)
                          FROM predictions
                          WHERE user_id = ?
                          GROUP BY decision;''', [clsf_id])

        for row in cursor:
            print row[0], row[1]


def delete_decisions():
    """
    Delete all decisions and related records
    """
    with sqlite3.connect(db_path) as db:
        # need to return dictionary so it matches csv stuff
        cursor = db.cursor()
        cursor.execute('DELETE FROM predictions;')
        cursor.execute('DELETE FROM decisions;')
        cursor.execute('DELETE FROM classifier_data;')
        cursor.execute('DELETE FROM classifier_data_balanced;')
        cursor.execute('DELETE FROM users WHERE type = "classifier";')
        cursor.execute('''UPDATE relations
                          SET true_rel = NULL
                          WHERE rel_id IN (SELECT rel_id
                                           FROM relations NATURAL JOIN sentences
                                           WHERE sentences.source = 'pubmed');''')


def update():
    """
    To be called from the server
    """
    update_correct_classifications()
    classify_remaining(optimise_params=False, no_biotext=False)


if __name__ == '__main__':
    update_correct_classifications()
    #classify_remaining(optimise_params=False, no_biotext=False)
    #count_true_false_predictions()
    #delete_decisions()
