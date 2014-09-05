import csv
import os


def single_sentence_relations():
    """
    Extract all sentences containing both drug and company ie our easy true relations
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/entities_marked_all.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/single_sentence_relations.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['DRUGS', 'COMPANIES', 'SENTENCE', 'SOURCE_ID', 'SENT_NUM', 'OTHER',
                                                  'POS_TAGS', 'D_CHUNKS', 'C_CHUNKS', 'O_CHUNKS', 'CHUNKS'],
                                        delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # need to use eval to get dicts back, this is horrible must find better method than CSV for everything
                if len(eval(row['DRUGS'])) > 0 and len(eval(row['COMPANIES'])) > 0:
                    csv_writer.writerow(row)

    # TODO write everything to one file with yes or no class attribute? This may avoid duplicates?
    print 'Written to single_sentence_relations.csv'


def single_sentence_non_relations():
    """
    Extract all sentences containing two or more 'other' entities
    Problem going to arise when the entities have multiple names
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/entities_marked_all.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/single_sentence_non_relations.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['OTHER', 'SENTENCE', 'SOURCE_ID', 'SENT_NUM', 'DRUGS', 'COMPANIES',
                                                  'POS_TAGS', 'D_CHUNKS', 'C_CHUNKS', 'O_CHUNKS', 'CHUNKS'],
                                        delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                # need to use eval to get dicts back, this is horrible must find better method than CSV for everything
                # TODO better generation of other entity pairs - this is very limited
                # for now just using those with a single pair of entities
                if len(eval(row['OTHER'])) == 2:
                    csv_writer.writerow(row)

    print 'Written to single_sentence_non_relations.csv'


def get_tags_between(all_tags, start, finish, element):
    """
    Return string for all elements in tags between start and finish
    """
    tags = all_tags[start:finish]
    # can put everything in lowercase in weka, probably better to do it there
    return '"' + ' '.join([tag[element] for tag in tags]) + '"'


def find_closest_pair(dict1, dict2, e1, e2):
    """
    Find closest pair of the two given entities and distance between them
    """
    # generate all pairs of indices, first need to unzip the dictionary values to only consider the indices
    try:
        pairs = [(x, y) for x in dict1[e1] for y in dict2[e2]]
    except:
        # print relevant info when an exception is caught, easier to debug
        print e1
        print e2
        print dict1
        print dict2
        return None, None

    # TODO need to validate this assumption
    # find closest pair, assuming these will have the relation between them
    distances = [abs(x[0] - y[0]) for (x, y) in pairs]
    # this is a kind of hacky way to deal with one entity whose prefix is a 'different' entity
    distances = [d for d in distances if d != 0]
    # if there is no actual pair of entities ie both are the same entity
    if len(distances) == 0:
        return None, None

    # calculate minimum distance and find closest pair of entities
    min_dist = min(distances)
    closest_pair = sorted(pairs[distances.index(min_dist)])
    return min_dist, closest_pair


def word_features(tags, closest_pair):
    """
    Extract features based on the words and tags between the entities
    """
    # starting index is index of first word + length of entity - 1
    # don't need to subtract the 1 however?
    e1_end = closest_pair[0][0] + closest_pair[0][1]
    e2_start = closest_pair[1][0]

    words = get_tags_between(tags, e1_end, e2_start, 0)
    pos_tags = get_tags_between(tags, e1_end, e2_start,  1)
    stem_tags = get_tags_between(tags, e1_end, e2_start, 2)

    return [words, stem_tags, pos_tags]


def generate_word_feature_vector(tags, sent_num, dict1, dict2, e1, e2):
    """
    Generate feature vector for given entity pair
    Return None if the entity pair should not have been selected, if one is prefix of the other
    """
    # create feature vector with first attribute - where the sentence is in the text
    f_vector = [sent_num]

    # find closest pair of the entities and return
    min_dist, closest_pair = find_closest_pair(dict1, dict2, e1, e2)
    # if there is no actual pair of entities ie both are the same entity
    if min_dist is None:
        return None
    # append word gap to feature vector
    f_vector.append(min_dist)

    # append word features to feature vector
    f_vector.extend(word_features(tags, closest_pair))

    return f_vector


def generate_chunk_feature_vector(chunk_tags, dict1, dict2, e1, e2):
    """
    Generate chunk feature vector for given entity pair
    Can assume here that the entities will be found since they exist in the tags part
    """
    # find closest pair of the entities and return
    min_dist, closest_pair = find_closest_pair(dict1, dict2, e1, e2)
    # if there is no actual pair of entities ie both are the same entity
    if min_dist is None:
        return None

    e1_end = closest_pair[0][0] + closest_pair[0][1]
    e2_start = closest_pair[1][0]
    #print chunk_tags
    #print chunk_tags[e1_end:e2_start]
    phrase_beginnings = [c[2] for c in chunk_tags[e1_end:e2_start] if c[2] is not None and c[2][0] == 'B']
    phrases = '"' + '-'.join([p[2:] for p in phrase_beginnings]) + '"'

    nps = sum(1 for p in phrase_beginnings if p == 'B-NP')
    vps = sum(1 for p in phrase_beginnings if p == 'B-VP')
    pps = sum(1 for p in phrase_beginnings if p == 'B-PP')

    # append word features to feature vector
    f_vector = [phrases, nps, vps, pps]

    return f_vector


def generate_true_set():
    """
    Create list of feature vectors for given entity pairs
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/single_sentence_relations.csv'))

    feature_vectors = []

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            drug_dict = eval(row['DRUGS'])
            comp_dict = eval(row['COMPANIES'])
            tags = eval(row['POS_TAGS'])

            drug_chunk_dict = eval(row['D_CHUNKS'])
            comp_chunk_dict = eval(row['C_CHUNKS'])
            chunk_tags = eval(row['CHUNKS'])

            # consider all drug-company pairs
            for drug in drug_dict.keys():
                for comp in comp_dict.keys():

                    # generate word features
                    f_vector = generate_word_feature_vector(tags, row['SENT_NUM'], drug_dict, comp_dict, drug, comp)

                    # if f vector is None then the entity pair should not be taken into consideration
                    if f_vector is not None:
                        # generate chunk features
                        chunk_f_vector = generate_chunk_feature_vector(chunk_tags, drug_chunk_dict, comp_chunk_dict,
                                                                       drug, comp)
                        if chunk_f_vector is not None:
                            f_vector.extend(chunk_f_vector)
                            # add class label
                            f_vector.append('yes')
                            feature_vectors.append(f_vector)

    return feature_vectors


def generate_false_set():
    """
    Create list of feature vectors for given entity pairs, these are the non-related for now
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/single_sentence_non_relations.csv'))

    feature_vectors = []

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            other_dict = eval(row['OTHER'])
            tags = eval(row['POS_TAGS'])
            chunk_dict = eval(row['O_CHUNKS'])
            chunk_tags = eval(row['CHUNKS'])

            # need to use range loops here since want unique pairs from the lists
            # TODO nicer way to do this?
            for i in xrange(len(other_dict.keys())):
                for j in xrange(i, len(other_dict.keys())):

                    e1, e2 = other_dict.keys()[i], other_dict.keys()[j]
                    f_vector = generate_word_feature_vector(tags, row['SENT_NUM'], other_dict, other_dict, e1, e2)

                    if f_vector is not None:
                        # generate chunk features
                        chunk_f_vector = generate_chunk_feature_vector(chunk_tags, chunk_dict, chunk_dict,
                                                                       e1, e2)
                        if chunk_f_vector is not None:
                            f_vector.extend(chunk_f_vector)
                            # add class label
                            f_vector.append('no')
                            feature_vectors.append(f_vector)

    return feature_vectors


def generate_feature_vectors():
    """
    Step 3 in the pipeline so far...
    Find entity pairs for true and false examples and create feature vectors
    """
    single_sentence_relations()
    single_sentence_non_relations()
    # these are called from weka.py
    #generate_true_set()
    #generate_false_set()


if __name__ == '__main__':
    generate_feature_vectors()
