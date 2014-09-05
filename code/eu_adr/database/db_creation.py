import sqlite3
import csv


def initial_setup(db_name, new=True):
    """
    Set up database based on preprocessed sentences from existing corpora
    """
    create_tables(db_name, new)
    populate_sentences(db_name)
    populate_relations(db_name)
    populate_users(db_name)
    clean_biotext_relations(db_name)


def create_tables(db_name, new=True):
    """
    Create tables to hold the data
    """
    with sqlite3.connect(db_name) as db:
        cursor = db.cursor()

        # drop the tables if overwriting existing db
        if not new:
            cursor.execute('DROP TABLE sentences;')
            cursor.execute('DROP TABLE relations;')
            cursor.execute('DROP TABLE users;')
            cursor.execute('DROP TABLE decisions;')
            cursor.execute('DROP TABLE predictions;')
            cursor.execute('DROP TABLE temp_sentences')
            cursor.execute('DROP TABLE classifier_data')

        # table for sentences themselves
        cursor.execute('''CREATE TABLE sentences(sent_id INTEGER PRIMARY KEY,
                                                 pubmed_id INTEGER,
                                                 sent_num INTEGER,
                                                 sentence TEXT,
                                                 source TEXT);''')

        # table for the relations
        cursor.execute('''CREATE TABLE relations(rel_id INTEGER PRIMARY KEY,
                                                 sent_id INTEGER,
                                                 true_rel BOOLEAN,
                                                 bad_ner BOOLEAN,
                                                 entity1 TEXT,
                                                 type1 TEXT,
                                                 start1 INTEGER,
                                                 end1 INTEGER,
                                                 entity2 TEXT,
                                                 type2 TEXT,
                                                 start2 INTEGER,
                                                 end2 INTEGER,
                                                 before_tags TEXT,
                                                 between_tags TEXT,
                                                 after_tags TEXT,
                                                 FOREIGN KEY(sent_id) REFERENCES sentences);''')

        # table for annotators
        cursor.execute('''CREATE TABLE users(user_id INTEGER PRIMARY KEY,
                                             name TEXT,
                                             type TEXT);''')

        # table for annotators decision
        cursor.execute('''CREATE TABLE decisions(decision_id INTEGER PRIMARY KEY,
                                                 rel_id INTEGER,
                                                 user_id INTEGER,
                                                 decision INTEGER,
                                                 reason TEXT,
                                                 FOREIGN KEY(rel_id) REFERENCES relations,
                                                 FOREIGN KEY(user_id) REFERENCES users);''')

        # table for the classifiers' predictions
        cursor.execute('''CREATE TABLE predictions(prediction_id INTEGER PRIMARY KEY,
                                                   rel_id INTEGER,
                                                   user_id INTEGER,
                                                   decision INTEGER,
                                                   confidence_value REAL,
                                                   FOREIGN KEY(rel_id) REFERENCES relations,
                                                   FOREIGN KEY(user_id) REFERENCES users);''')

        # table for new sentences containing possible relations
        cursor.execute('''CREATE TABLE temp_sentences(sent_id INTEGER,
                                                      entity_dict TEXT,
                                                      FOREIGN KEY(sent_id) REFERENCES sentences);''')

        # table for classifier training data
        cursor.execute('''CREATE TABLE classifier_data(clsf_id INTEGER,
                                                       training_rel INTEGER,
                                                       FOREIGN KEY(clsf_id) REFERENCES users(user_id),
                                                       FOREIGN KEY(training_rel) REFERENCES relations(rel_id));''')

        # table for balanced data really used for training
        cursor.execute('''CREATE TABLE classifier_data_balanced(clsf_id INTEGER,
                                                                training_rel INTEGER,
                                                       FOREIGN KEY(clsf_id) REFERENCES users(user_id),
                                                       FOREIGN KEY(training_rel) REFERENCES relations(rel_id));''')


def populate_sentences(db_name):
    """
    Populate the sentences table with initial set of sentences from biotext and eu-adr corpora
    """
    with open('../csv/tagged_sentences_NEW.csv', 'rb') as f_in:
        csv_reader = csv.DictReader(f_in, delimiter=',')
        pid = 0
        sent_num = 0

        with sqlite3.connect(db_name) as db:
            cursor = db.cursor()

            for row in csv_reader:
                # this isn't a great way to do it, relies on spreadsheet being ordered 
                if row['pid'] != pid or row['sent_num'] != sent_num:
                    # set the source, this should make it easier to query new records later
                    if eval(row['pid']) < 1000:
                        src = 'biotext'
                    else:
                        src = 'eu-adr'
                    cursor.execute('INSERT INTO sentences VALUES (NULL, ?, ?, ?, ?);',
                                   # sentences saved in utf-8 but sqlite wants unicode -> need to decode
                                   (row['pid'], row['sent_num'], row['sentence'].decode('utf-8'), src))
                pid = row['pid']
                sent_num = row['sent_num']


def populate_relations(db_name):
    """
    Populate the relations table with set of 'correctly' annotated relations
    """
    with open('../csv/tagged_sentences_NEW.csv', 'rb') as f_in:
        csv_reader = csv.DictReader(f_in, delimiter=',')

        with sqlite3.connect(db_name) as db:
            cursor = db.cursor()

            for row in csv_reader:
                # retrieve sentence ID for this sentence
                cursor.execute('''SELECT sent_id
                                  FROM sentences
                                  WHERE pubmed_id = ? AND sent_num = ?''', (row['pid'], row['sent_num']))
                try:
                    sent_id = cursor.fetchone()[0]
                except:
                    print 'pid =', row['pid']
                    print 'sent_num =', row['sent_num']
                    return 0

                # need to set bool values as 0 or 1 for sqlite
                if eval(row['true_relation']):
                    true_rel = 1
                else:
                    true_rel = 0

                cursor.execute('INSERT INTO relations VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);',
                               (sent_id,
                                true_rel,
                                0,  # this is the bad NER field, false for the original relations
                                row['e1'].decode('utf-8'),
                                row['type1'],
                                row['start1'],
                                row['end1'],
                                row['e2'].decode('utf-8'),
                                row['type2'],
                                row['start2'],
                                row['end2'],
                                row['before_tags'].decode('utf-8'),
                                row['between_tags'].decode('utf-8'),
                                row['after_tags'].decode('utf-8')))


def populate_users(db_name):
    """
    Populate the decisions table to record 'correct' annotations from the corpora
    """
    with sqlite3.connect(db_name) as db:
        cursor = db.cursor()

        cursor.execute('''INSERT INTO users
                                 VALUES (NULL, 'Douglas', 'testing'), (NULL, 'Andrew', 'user');''')


def clean_biotext_relations(db_name):
    """
    Remove biotext relations deemed unhelpful
    """
    with sqlite3.connect(db_name) as db:
        cursor = db.cursor()
        cursor.execute('''DELETE FROM relations
                          WHERE rel_id IN (SELECT rel_id
                                           FROM relations NATURAL JOIN sentences
                                           WHERE source = 'biotext' AND
                                                 (entity1 LIKE '%therapy%' OR entity1 LIKE '%therapi%' OR entity1 LIKE
                                                 '%surgery%' OR entity1 LIKE '%surgical%' OR entity1 LIKE '%treatment%'
                                                  OR entity2 LIKE '%therapy%' OR entity2 LIKE '%therapi%' OR entity2
                                                  LIKE  '%surgery%' OR entity2 LIKE '%surgical%' OR entity2 LIKE
                                                  '%treatment%'));''')

if __name__ == '__main__':
    #create_tables()
    #populate_sentences()
    #populate_relations()
    #populate_decisions()
    #initial_setup()
    #create_temp_sentences()
    #create_classifier_table()
    #clean_biotext_relations()
    #boom()
    initial_setup('relex_new.db')
    #clean_biotext_relations('relex_new.db')
