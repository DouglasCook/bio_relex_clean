import sqlite3

from tagger import TaggerChunker

from reuters_NER import ontologies_api_demo as ner

db_path = 'database/test.db'


def relevant_into_temp():
    """
    Populate temporary table with relevant sentences from pubmed query
    This is a very slow process so NEED TO DO IT ON THE SERVER!
    """
    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        cursor.execute('''SELECT *
                          FROM sentences
                          WHERE source = 'pubmed';''')
        # need to fetch all here since we are using same cursor later for insert query
        sentences = cursor.fetchall()

        for row in sentences:
            # call the reuters ner
            raw_xml = ner.runOntologiesSearch(row['sentence'])
            entity_dict = ner.summariseXml(raw_xml)

            # need at least two entities for a relation
            if len(entity_dict) > 1:
                disorder_present = False
                treatment_present = False

                # check all entities that have been tagged
                for key in entity_dict:
                    if entity_dict[key][0] == 'Indication':
                        disorder_present = True
                    elif entity_dict[key][0] == 'Drug':
                        treatment_present = True

                    # if there is at least one action and one indication the sentence is relevant
                    if disorder_present and treatment_present:
                        cursor.execute('''INSERT INTO temp_sentences
                                                 VALUES (?, ?);''',
                                       (row['sent_id'], str(entity_dict)))
                        # print just to see progress
                        print row['sent_id']
                        # committing here will slow things down but means can break and still have SOME data
                        # slow down is negligible in grand scheme of things
                        db.commit()
                        break


def split_sentence(sent, start1, end1, start2, end2):
    """
    Split a sentence into before, between and after sections
    """
    # not interested in leading or trailing spaces so strip them off
    return sent[:start1].strip(), sent[end1+1:start2].strip(), sent[end2+1:].strip()


def add_to_relations():
    """
    Extract potential relations from relevant sentences, tag them and add to relations table
    """
    tagger = TaggerChunker()

    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cursor = db.cursor()
        # take everything from relevant sentences that has not already been checked for relations
        cursor.execute('''SELECT temp_sentences.*, sentences.sentence
                          FROM temp_sentences NATURAL JOIN sentences
                          WHERE sent_id NOT IN (SELECT DISTINCT(sent_id)
                                                FROM relations);''')
        sentences = cursor.fetchall()

        for row in sentences:
            sent_id = row['sent_id']
            entity_dict = eval(row['entity_dict'])
            # create lists for disorders and treatments
            disorders = [k for k in entity_dict if entity_dict[k][0] == 'Indication']
            treatments = [k for k in entity_dict if entity_dict[k][0] == 'Drug']

            # loop through all pairs
            for treat in treatments:
                treat_dict = entity_dict[treat]
                for disorder in disorders:
                    dis_dict = entity_dict[disorder]

                    # split sentence dependent on which entity appears first
                    if treat_dict[1] < dis_dict[1]:
                        e1, e2 = treat, disorder
                        type1, type2 = 'Drug', 'Disorder'
                        start1, start2 = treat_dict[1], dis_dict[1]
                        end1, end2 = treat_dict[2], dis_dict[2]
                    else:
                        e1, e2 = disorder, treat
                        type1, type2 = 'Disorder', 'Drug'
                        start1, start2 = dis_dict[1], treat_dict[1]
                        end1, end2 = dis_dict[2], treat_dict[2]

                    before, between, after = split_sentence(row['sentence'], start1, end1, start2, end2)
                    # tag and chunk the parts of sentence
                    bef_chunks, bet_chunks, aft_chunks = tagger.pos_and_chunk_tags(row['sentence'], before,
                                                                                       between, e1, e2)

                    # TODO do I need to catch the exceptions here, don't want it to crash?
                    try:
                        cursor.execute('''INSERT INTO relations
                                          VALUES (NULL, ?, NULL, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                       (sent_id, e1, type1, start1, end1, e2, type2, start2, end2, str(bef_chunks),
                                        str(bet_chunks), str(aft_chunks)))
                    except:
                        pass


if __name__ == '__main__':
    #test()
    relevant_into_temp()
    add_to_relations()
