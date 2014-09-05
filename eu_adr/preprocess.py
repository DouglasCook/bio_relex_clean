import os
import sys
import csv
import pickle

from tagger import TaggerChunker


def relations_to_dict():
    """
    Put all relations from EU-ADR corpus into dictionary to pickle
    """
    # load pubmed ids
    pubmed_ids = pickle.load(open('eu-adr_ids.p', 'rb'))
    basepath = os.path.dirname(__file__)

    relation_dict = {}

    for pid in pubmed_ids:
        f = os.path.abspath(os.path.join(basepath, '..', '..', 'data', 'euadr_corpus', pid + '.csv'))
        r_dict = {'start1': [], 'end1': [], 'type1': [], 'start2': [], 'end2': [], 'type2': [], 'rel_type': [],
                  'true_relation': []}

        with open(f, 'rb') as csv_in:
            # no point in using dict reader since there are no headings
            csv_reader = csv.reader(csv_in, delimiter='\t')

            for row in csv_reader:
                # if the row describes a relation add details to dict
                if row[2] == 'relation':
                    # want to split into start and end points for ease of use later
                    indices = row[7].split(':')
                    r_dict['start1'].append(int(indices[0]))
                    r_dict['end1'].append(int(indices[1]))
                    r_dict['type1'].append(row[0].split('-')[0])

                    indices = row[8].split(':')
                    r_dict['start2'].append(int(indices[0]))
                    r_dict['end2'].append(int(indices[1]))
                    r_dict['type2'].append(row[0].split('-')[1])

                    r_dict['true_relation'].append(row[1])
                    r_dict['rel_type'].append(row[10])

        # apparently some of the files contain no relations so need to check that here
        # can just use 'if object' instead of 'len(object) > 0'
        if r_dict['start1']:
            # now we need to order them by location since the spreadsheet is not in order
            sorter = zip(r_dict['start1'], r_dict['end1'], r_dict['type1'], r_dict['start2'], r_dict['end2'],
                         r_dict['type2'], r_dict['rel_type'], r_dict['true_relation'])
            sorter.sort()
            (r_dict['start1'], r_dict['end1'], r_dict['type1'], r_dict['start2'], r_dict['end2'], r_dict['type2'],
             r_dict['rel_type'], r_dict['true_relation']) = zip(*sorter)

            relation_dict[pid] = r_dict

    # pickle it
    # print relation_dict['16950808']
    pickle.dump(relation_dict, open('relation_dict.p', 'wb'))


def create_output_row(relations, row, i, length):
    """
    Generate row to be written to csv with entities located
    """
    # TODO sort out the accents
    # can use unidecode module but this creates problems with other sentences eg 16950808 Sjorsen's disease
    sent = unicode(row['text'], 'utf-8')

    # want first first entity to be that which appears first in text
    if relations['start1'][i] < relations['start2'][i]:
        start1 = relations['start1'][i] - length
        end1 = relations['end1'][i] - length
        type1 = relations['type1'][i]
        start2 = relations['start2'][i] - length
        end2 = relations['end2'][i] - length
        type2 = relations['type2'][i]
    else:
        start1 = relations['start2'][i] - length
        end1 = relations['end2'][i] - length
        type1 = relations['type2'][i]
        start2 = relations['start1'][i] - length
        end2 = relations['end1'][i] - length
        type2 = relations['type1'][i]

    # the entities themselves
    e1 = sent[start1:end1]
    e2 = sent[start2:end2]

    # relation information
    rel = relations['true_relation'][i]
    rel_type = relations['rel_type'][i]

    # parts of the sentence
    between = sent[end1 + 1:start2].strip()
    before = sent[:start1].strip()
    after = sent[end2 + 1:].strip()

    # need to re-encode everything as utf-8 ffs
    return [row['id'],
            row['sent_num'],
            rel,
            rel_type,
            e1,
            e2,
            type1,
            type2,
            start1,
            end1,
            start2,
            end2,
            sent,
            before,
            between,
            after]


def create_input_file():
    """
    Set up input file with only those sentences containing relations
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, 'csv/sentences.csv'))
    file_out = os.path.abspath(os.path.join(basepath, 'csv/relevant_sentences.csv'))

    relation_dict = pickle.load(open('relation_dict.p', 'rb'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:

            # set up columns here so they are easier to play around with
            cols = ['pid',
                    'sent_num',
                    'true_relation',
                    'rel_type',
                    'e1',
                    'e2',
                    'type1',
                    'type2',
                    'start1',
                    'end1',
                    'start2',
                    'end2',
                    'sentence',
                    'before',
                    'between',
                    'after']
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, cols, delimiter=',')
            csv_writer.writerow(cols)

            length, i = 0, 0

            # loop through all sentences
            for row in csv_reader:
                # check that the key exists
                if row['id'] in relation_dict:
                    # if we are at start of next text reset counters
                    if row['sent_num'] == '0':
                        length, i = 0, 0

                        # get the relation info for this sentence
                        relations = relation_dict[row['id']]

                    # if not reached the end of these relations and
                    # if the next entity is in this sentence
                    while i < len(relations['start1']) and relations['start1'][i] < length + len(row['text']):
                        csv_writer.writerow(create_output_row(relations, row, i, length))
                        i += 1

                    # add length of sentence to total, need to add one for the missing space between sentences
                    length += len(row['text']) + 1


def drug_disorder_only():
    """
    Extract drug-disorder relations from relevant sentences and leave the rest
    """
    with open('csv/relevant_sentences.csv', 'rb') as csv_in:
        with open('csv/drug_disorder_only.csv', 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            cols = ['pid',
                    'sent_num',
                    'true_relation',
                    'e1',
                    'e2',
                    'type1',
                    'type2',
                    'start1',
                    'end1',
                    'start2',
                    'end2',
                    'sentence',
                    'before',
                    'between',
                    'after']
            csv_writer = csv.DictWriter(csv_out, cols, delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # if its a drug disorder relation
                if row['type1'] in ['Drug', 'Disorder'] and row['type2'] in ['Drug', 'Disorder']:
                    # remove extra fields not obtainable from biotext
                    row.pop('rel_type')
                    csv_writer.writerow(row)


def tagging(filename, new_file=False):
    """
    Tags and chunk words between the two entities
    """
    # set filepath to input
    file_in = 'csv/' + filename
    file_out = 'csv/tagged_sentences_TREEBANK_CHUNKS.csv'

    chunker = TaggerChunker()

    # want to append or write over depending on situation
    if new_file:
        mode = 'wb'
    else:
        mode = 'ab'

    with open(file_in, 'rb') as csv_in:
        with open(file_out, mode) as csv_out:
            # set columns here so they can be more easily changed
            cols = ['pid',
                    'sent_num',
                    'true_relation',
                    'e1',
                    'e2',
                    'type1',
                    'type2',
                    'start1',
                    'end1',
                    'start2',
                    'end2',
                    'sentence',
                    'before_tags',
                    'between_tags',
                    'after_tags']
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, cols, delimiter=',', extrasaction='ignore')

            # only write header when creating new file
            if new_file:
                csv_writer.writeheader()

            for row in csv_reader:
                # display progress bar
                sys.stdout.write('.')
                sys.stdout.flush()

                # tag and chunk the parts of sentence
                bef, bet, aft = chunker.pos_and_chunk_tags(row['sentence'], row['before'], row['between'],
                                                               row['e1'], row['e2'])
                row.update({'before_tags': bef})
                row.update({'between_tags': bet})
                row.update({'after_tags': aft})

                csv_writer.writerow(row)


if __name__ == '__main__':
    #abstracts_to_csv()
    #relations_to_dict()
    #create_input_file()
    #drug_disorder_only()
    tagging('drug_disorder_only.csv', new_file=True)
    tagging('biotext.csv', new_file=False)
