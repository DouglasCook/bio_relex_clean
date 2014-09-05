import os

import feature_extraction


def create_arff(filepath, relation, attributes):
    """
    Create arff file for use in WEKA, don't worry about comments or anything for now
    Requires lists of lists for attributes and data
    Dictionary seems like it could be useful to use here but has no order so maybe better to stick with list?
    """
    with open(filepath, 'w') as f:
        # create relation title
        f.write('@RELATION ' + relation)
        f.write('\n' * 2)

        # now add attributes and their types
        for [att, att_type] in attributes:
            f.write('@ATTRIBUTE ' + att + ' ' + att_type + '\n')
        f.write('\n@DATA\n')


def add_arff_data(filepath, data):
    """
    Add data to given arff file
    """
    with open(filepath, 'a') as f:
        # add data to data section
        for vector in data:
            vector = [str(v) for v in vector]
            f.write(', '.join(vector) + '\n')


def write_file(file_name):
    """
    Step 4 in the pipeline so far...
    Write the weka file
    """
    # want to be able to save to different files easily
    basepath = os.path.dirname(__file__)
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/WEKA/' + file_name + '.arff'))

    # write the header stuff
    create_arff(file_out, 'drug_company', [['sent_num', 'NUMERIC'],
                                           ['word_gap', 'NUMERIC'],
                                           ['words', 'STRING'],
                                           ['stems', 'STRING'],
                                           ['pos_tags', 'STRING'],
                                           ['phrase_path', 'STRING'],
                                           ['num_NPs', 'NUMERIC'],
                                           ['num_VPs', 'NUMERIC'],
                                           ['num_PPs', 'NUMERIC'],
                                           ['true_relation', '{yes, no}']])
    # add the data
    add_arff_data(file_out, feature_extraction.generate_true_set())
    add_arff_data(file_out, feature_extraction.generate_false_set())


if __name__ == '__main__':
    file_name = raw_input('Enter file name ')
    write_file(file_name)
