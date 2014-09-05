import os
import csv
import re
import itertools        # this is used to loop over elements of consecutive lists


def entity_indices(entities, tags):
    """
    Given entities return indices in tags where their first words are found
    Return dictionary with entities as keys and indices as values
    """
    entity_dict = {}
    for entity in entities:
        # TODO fix this method, sometimes first word makes sense but sometimes identifies incorrect entities
        # locate first word if entity is made of multiple words
        words = entity.split()
        head_word = words[0]
        # underscores are used in the tokens so need to replace before searching
        head_word = head_word.replace('-', '_')
        tokens = [tag[0] for tag in tags]

        # generate list off indices for all occurrences of this entity
        indices = [i for i, x in enumerate(tokens) if x == head_word]
        # TODO does casting everything as lower case give correct results?
        #indices = [i for i, x in enumerate(tags) if x[0].lower() == head_word.lower()]

        # also want to calculate length of word
        lengths = []
        for i in indices:
            # if entity is a single word need to deal with it here before the j loop
            if len(words) == 1:
                lengths.append([i, 1])
            # otherwise calculate length of the word with this horrible loop...
            else:
                for j in xrange(1, len(words)):
                    # if we have reached end of text need to avoid index out of range
                    if i + j >= len(tokens):
                        lengths.append([i, max(j - 1, 1)])
                        break
                    # if the next word doesn't match then length is j
                    if tokens[i + j] != words[j]:
                        lengths.append([i, j])
                        break
                    # if all words are matched then length is len(words) ie j + 1
                    if j == len(words) - 1:
                        lengths.append([i, j + 1])

        if len(lengths) > 0:
            entity_dict[entity] = lengths

    return entity_dict


def other_entity_indices(entities, drug_dict, comp_dict, tags):
    """
    Given entities return indices in tags where their first words are found
    Return dictionary with entities as keys and indices as values
    """
    entity_dict = {}
    for entity in entities:
        # underscores are used in the tokens so need to replace before searching
        entity = entity.replace('-', '_')
        words = entity.split()
        # want to record length of entity as well as index
        e_length = len(words)
        tokens = [tag[0] for tag in tags]
        indices = []

        # search for whole entity in tokens
        for i in xrange(len(tokens)):
            if tokens[i:i+len(words)] == words:
                indices.append([i, e_length])

        # TODO this seems to be working but not completely solid
        if len(indices) > 0:
            # the chain creates an iterable by iterating over all iterable arguments (in this case the lists)
            # then need to extract the indices only
            d_chain = [d[0] for d in itertools.chain(*drug_dict.values())]
            c_chain = [c[0] for c in itertools.chain(*comp_dict.values())]

            # if the entity doesn't share an index with drug or company add it
            # check single occurrence of 'other' against all occurrences of drugs and companies
            if indices[0][0] not in d_chain and indices[0][0] not in c_chain:
                entity_dict[entity] = indices

    return entity_dict


def entities_only(filename, no_orgs):
    """
    Extract all named entities from the given text file
    """
    with open(filename, 'r') as f_in:
        text = f_in.read()
        # regex to match words within tags
        if no_orgs:
            # below will not match any organisations, may be better for now for generating false examples
            entities = re.findall('<.{,8}>.+?</', text)
        else:
            entities = re.findall('<.{,12}>.+?</', text)

        # strip tags and remove duplicates
        entities = set([x[x.find('>')+1:x.rfind('<')] for x in entities])

    return entities


def drug_and_company_entities():
    """
    Locate named drugs and companies, indexed by word
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/sentences_POS.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/entities_marked.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['SOURCE_ID', 'SENT_NUM', 'SENTENCE', 'DRUGS', 'COMPANIES',
                                                  'POS_TAGS', 'D_CHUNKS', 'C_CHUNKS', 'CHUNKS'],
                                        delimiter=',', extrasaction='ignore')
            csv_writer.writeheader()

            for row in csv_reader:
                drugs = eval(row['DRUGS'])
                comps = eval(row['COMPANIES'])

                # find indices for drugs and companies mentioned in the row
                tags = eval(row['POS_TAGS'])
                drug_dict = entity_indices(drugs, tags)
                comp_dict = entity_indices(comps, tags)
                row.update({'DRUGS': drug_dict, 'COMPANIES': comp_dict})

                # do the same for chunk tags - tokens are different so need to redo it to get correct indices
                # maybe want to strip out anything that isn't beginning of a phrase before this?
                chunks = eval(row['CHUNKS'])
                drug_dict = entity_indices(drugs, chunks)
                comp_dict = entity_indices(comps, chunks)
                row.update({'D_CHUNKS': drug_dict, 'C_CHUNKS': comp_dict})

                # remove this field, think pop is the only way to do it
                # TODO what do I actually need this for in the first place?
                row.pop('NO_PUNCT')
                csv_writer.writerow(row)

    print 'Written to entities_marked.csv'


def other_entities(no_orgs):
    """
    Locate named entities tagged by Stanford NER tool
    Text file must be created via bash script for now, really not a good way to do it

    If no_orgs is True then no organisations will be considered for the 'other' entities
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/entities_marked.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/entities_marked_all.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['SOURCE_ID', 'SENT_NUM', 'SENTENCE', 'DRUGS', 'COMPANIES', 'OTHER',
                                                  'POS_TAGS', 'D_CHUNKS', 'C_CHUNKS', 'O_CHUNKS', 'CHUNKS'],
                                        delimiter=',', extrasaction='ignore')
            csv_writer.writeheader()

            for row in csv_reader:
                # extract tagged entities from preprocessed file
                ne_filepath = os.path.abspath(os.path.join(basepath, '..', 'reuters/stanford_input/named_entities'))
                entities = entities_only(ne_filepath + '/' + row['SOURCE_ID'] + '.txt', no_orgs)

                # find entities in POS tags
                entities_dict = other_entity_indices(entities, eval(row['DRUGS']), eval(row['COMPANIES']),
                                                     eval(row['POS_TAGS']))
                row.update({'OTHER': entities_dict})

                # find entities in chunk tags
                entities_dict = other_entity_indices(entities, eval(row['D_CHUNKS']), eval(row['C_CHUNKS']),
                                                     eval(row['CHUNKS']))
                row.update({'O_CHUNKS': entities_dict})

                csv_writer.writerow(row)

    print 'Written to entities_marked_all.csv'


def extract_all_entities(no_orgs):
    """
    Step 2 in the pipeline so far...
    Locate all named entities within the sentences (this is reliant on bash script having run first)
    """
    drug_and_company_entities()
    other_entities(no_orgs)


if __name__ == '__main__':
    extract_all_entities(True)
    #stanford_input_split_sentences()
