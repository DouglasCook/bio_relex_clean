import random
import sys
import os
import sqlite3

from flask import Flask
from flask import render_template
from flask import request
from flask import session
from flask import redirect

import utility
import retraining

# TODO exception catching / error redirects?
app = Flask(__name__)
app.secret_key = os.urandom(24)
db_path = utility.build_filepath(__file__, '../database/relex.db')


@app.route('/')
@app.route('/index')
def user_selection():
    user_list = get_user_list()
    return render_template('login.html', users=user_list)


def get_user_list():
    """
    Return list of existing users for login page
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute("SELECT * FROM users WHERE type = 'user'")
    return [u for u in cursor]


@app.route('/login', methods=['POST'])
def login():
    """
    Store user in session and find next relation for them to annotate
    """
    # store user in session
    user_id = request.form['user']
    session['user_id'] = user_id

    # save how many user has done so far in session
    total_done(user_id)

    # save relation data in session
    select_relations(user_id)
    remaining_to_do(user_id)

    return redirect('/classify')


def select_relations(user_id):
    """
    Generate set of relations to be classified by user and save in session
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # TODO change query dependent on active learning method eg decision function
        """
        # THIS ORDERS RELATIONS BY THEIR DISTANCE FROM THE SEPARATING HYPERPLANE
        cursor.execute('''SELECT rel_id
                          FROM relations NATURAL JOIN predictions
                          WHERE relations.true_rel IS NULL AND
                                relations.bad_ner = 0 AND
                                predictions.user_id = (SELECT max(user_id)
                                                       FROM users
                                                       WHERE type = 'classifier') AND
                                relations.rel_id NOT IN (SELECT rel_id
                                                         FROM decisions
                                                         WHERE decisions.user_id = ?)
                           ORDER BY predictions.confidence_value;''', [user_id])

        """
        # THIS JUST TAKES EVERYTHING THAT HAS NOT BEEN HUMAN ANNOTATED
        cursor.execute('''SELECT rel_id
                          FROM relations
                          WHERE relations.true_rel IS NULL AND
                                relations.bad_ner = 0 AND
                                relations.rel_id NOT IN (SELECT rel_id
                                                         FROM decisions
                                                         WHERE decisions.user_id = ?);''', [user_id])

        """
        #USE THIS TO SEE WHAT HAS BEEN CLASSIFIED AS TRUE
        cursor.execute('''SELECT rel_id
                          FROM relations NATURAL JOIN predictions
                          WHERE relations.true_rel IS NULL AND
                                relations.bad_ner = 0 AND
                                predictions.decision = 1 AND
                                relations.rel_id NOT IN (SELECT rel_id
                                                         FROM decisions
                                                         WHERE decisions.user_id = ?);''', [user_id])
        """

        # create list of relations to classify to iterate through
        rels = [c[0] for c in cursor]
        # TODO shuffling is one possible strategy, in order is another, distance from support vectors is another
        # DO NOT shuffle when using active learning!
        random.shuffle(rels)
        session['rels_to_classify'] = rels
        session['number_rels'] = len(rels)
        session['next_index'] = 0


def remaining_to_do(user_id):
    """
    Calculate how many relations need to be classified before retraining
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # now want to count how many relations remain to be annotated before retraining
        # first count number of relations already classified
        cursor.execute('''SELECT count(rel_id)
                          FROM relations
                          WHERE true_rel IS NOT NULL''')
        # want to retrain once we have 5% more data - python 2 so will floor the result
        to_do = cursor.fetchone()[0]/20
        print 'to do', to_do

        # now count all classifications done since last retraining
        # do not include those unsure or bad NER since they tell the classifier nothing
        cursor.execute('''SELECT count(rel_id)
                          FROM decisions
                          WHERE user_id = ? AND
                                decision < 2 AND
                                rel_id NOT IN (SELECT rel_id
                                               FROM relations
                                               WHERE true_rel IS NOT NULL)''', [user_id])

        session['still_to_do'] = to_do - cursor.fetchone()[0]
        print 'still to do', session['still_to_do']


def total_done(user_id):
    """
    Count how many annotations have been done by this user so far
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # just count number of decisions user has made
        cursor.execute('''SELECT count(decision_id)
                          FROM decisions
                          WHERE user_id = ?''', [user_id])

        session['total_done'] = cursor.fetchone()[0]


@app.route('/classify')
def next_to_classify():
    """
    Display next relation to be classified
    """
    next_rel = session['rels_to_classify'][int(session['next_index'])]
    before, between, after, e1, e2, prediction, type1, link = return_relation(next_rel)
    total_so_far = session['total_done']
    to_do_before_retraining = session['still_to_do']

    # set radio button to pre check the classifiers prediction
    if prediction:
        pred = 'True'
        true_check = True
    else:
        pred = 'False'
        true_check = False

    # set correct colouring for drugs and disorders
    if type1 == 'Drug':
        drug_first = True
    else:
        drug_first = False

    return render_template('index.html', before=before, between=between, after=after, classification=pred,
                           e1=e1, e2=e2, true_check=true_check, drug_first=drug_first, link=link,
                           total_done=total_so_far, to_do=to_do_before_retraining)


def return_relation(rel_id):
    """
    Get potential relation from database
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        cursor.execute('''SELECT sentences.sentence,
                                 sentences.pubmed_id,
                                 relations.entity1,
                                 relations.type1,
                                 relations.start1,
                                 relations.end1,
                                 relations.entity2,
                                 relations.type2,
                                 relations.start2,
                                 relations.end2,
                                 relations.rel_id,
                                 predictions.decision
                          FROM sentences NATURAL JOIN relations
                                         NATURAL JOIN predictions
                          WHERE relations.rel_id = ? AND
                                predictions.user_id = (SELECT max(user_id)
                                                       FROM users
                                                       WHERE type = 'classifier');''', [rel_id])

        row = cursor.fetchone()

        # split sentence to render around entities
        before, between, after = utility.split_sentence(row['sentence'], row['start1'], row['end1'], row['start2'],
                                                        row['end2'])
        # build link to full abstract for context
        link = 'http://www.ncbi.nlm.nih.gov/pubmed/?term=%s' % row['pubmed_id']

        return before, between, after, row['entity1'], row['entity2'], row['decision'], row['type1'], link


@app.route('/save', methods=['POST'])
def record_decision():
    """
    Save annotators decision and redirect to next relation to be classified
    """
    # write decision to the database
    # remember to cast as int since sqlite doesn't enforce types
    decision = int(request.form['class'])
    store_decision(decision, request.form['reason'])

    # redirect to finished page if there are no remaining relations
    if session['next_index'] == session['number_rels'] - 1:
        return render_template('finished.html')

    # increment total done
    session['total_done'] += 1

    # only decrement counter if classification is true or false (not unsure or bad NER)
    if decision < 2:
        session['still_to_do'] -= 1
        print 'still to do', session['still_to_do']

    # if it's time to retrain, <= 0 to avoid potential bugs creeping in
    if session['still_to_do'] <= 0:
        # retrain and classify remaining
        retraining.update()
        # requery relations etc
        select_relations(session['user_id'])
        remaining_to_do(session['user_id'])
        return redirect('/classify')

    # increment next relation index
    session['next_index'] += 1

    return redirect('/classify')


def store_decision(classification, reason):
    """
    Record the annotators classification
    """
    rel_id = session['rels_to_classify'][int(session['next_index'])]
    user_id = session['user_id']

    with sqlite3.connect(db_path) as db:
        cursor = db.cursor()

        # if reason is given add it to the table
        if reason:
            cursor.execute('''INSERT into decisions
                              VALUES (NULL, ?, ?, ?, ?)''',
                           (rel_id, user_id, classification, reason))
        else:
            cursor.execute('''INSERT into decisions
                              VALUES (NULL, ?, ?, ?, NULL)''',
                           (rel_id, user_id, classification))

        # set bad NER to true if annotator believes it is
        # ideally this wouldn't happen here, it would need to be a majority decision so would be in retraining
        if classification == '2':
            cursor.execute('''UPDATE relations
                              SET bad_ner = 1
                              WHERE rel_id = ?''', [rel_id])


if __name__ == '__main__':
    # set command line arg to 1 to go live
    if int(sys.argv[1]) == 1:
        app.run(host='0.0.0.0', port=55100)
    else:
        app.run(debug=True)
