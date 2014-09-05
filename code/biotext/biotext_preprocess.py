import csv  # used for accessing data held in CSV format
import re  # regular expressions for extracting named entities

from eu_adr.app import utility
import preprocessing


def pos_tags():
    """ Create new CSV containing all relevant sentences """
    # set filepath to input
    filepath = utility.build_filepath(__file__, '../../data/biotext/sentences_with_roles_and_relations.txt')
    file_out = utility.build_filepath(__file__, '../biotext/sentences_POS.csv')
    print filepath
    print file_out

    with open(filepath, 'r') as f_in:
        with open(file_out, 'wb') as csv_out:
            csv_writer = csv.writer(csv_out, delimiter=',')

            csv_writer.writerow(['CLASS_TAG', 'SENTENCE', 'POS_TAGS'])

            for line in f_in:
                # split sentence and class tag and strip newline
                line = line.split('||')
                line[1] = line[1].rstrip()

                # only looking at treatment and disease sentences for now
                if line[1] == 'TREAT_FOR_DIS':
                    csv_writer.writerow([line[1], line[0], preprocessing.clean_and_tag_sentence(line[0])])


def entity_extraction():
    """
    Remove tags from sentences and record start and end points of entities
    """
    f_in = utility.build_filepath(__file__, 'csv/sentences.csv')
    f_out = utility.build_filepath(__file__, 'csv/sentences_entities.csv')

    with open(f_in, 'rb')as csv_in:
        with open(f_out, 'wb')as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
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
            csv_writer = csv.DictWriter(csv_out, fieldnames=cols, delimiter=',')
            csv_writer.writeheader()

            row_id = 0
            for row in csv_reader:
                # stupid spaces before fullstops
                sent = row[0].strip('.\s')

                # only want to deal with pairs of entities
                diseases = re.findall(r'<DIS>[^<]*', sent)
                treatments = re.findall(r'<TREAT>[^<]*', sent)
                if len(diseases) == 1 and len(treatments) == 1:

                    # if disease comes first
                    if sent.find(diseases[0]) < sent.find(treatments[0]):
                        sent, e1, start1, end1 = locate_entities('disease', sent)
                        sent, e2, start2, end2 = locate_entities('treatment', sent)
                        type1 = 'Disorder'
                        type2 = 'Drug'
                    else:
                        sent, e1, start1, end1 = locate_entities('treatment', sent)
                        sent, e2, start2, end2 = locate_entities('disease', sent)
                        type1 = 'Drug'
                        type2 = 'Disorder'

                    between = sent[end1 + 1:start2].strip()
                    before = sent[:start1].strip()
                    after = sent[end2 + 1:].strip()

                    dict_out = {'pid': row_id,
                                'sent_num': 0,  # this defaults to zero since they are only single sentences
                                'true_relation': 'True',
                                'e1': e1,
                                'e2': e2,
                                'type1': type1,
                                'type2': type2,
                                'start1': start1,
                                'end1': end1,
                                'start2': start2,
                                'end2': end2,
                                'sentence': sent,
                                'before': before,
                                'between': between,
                                'after': after}
                    csv_writer.writerow(dict_out)

                row_id += 1


def locate_entities(e_type, sentence):
    """
    Return entities in sentence and their indices based on type we are looking for
    """
    # find the entity
    if e_type == 'disease':
        match = re.search(r'<DIS>[^<]*', sentence)
    else:
        match = re.search(r'<TREAT>[^<]*', sentence)
    start = match.start()
    end = match.end()

    # slice up the sentence to extract features
    if e_type == 'disease':
        entity = sentence[start + 6:end - 1]
        sentence = sentence[:start] + sentence[start + 6:end] + sentence[end + 7:]
        end -= 7
    else:
        entity = sentence[start + 8:end - 1]
        sentence = sentence[:start] + sentence[start + 8:end] + sentence[end + 9:]
        end -= 9

    return sentence, entity, start, end


if __name__ == '__main__':
    pos_tags()
    #entity_extraction()
