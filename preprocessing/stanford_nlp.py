import os
import csv


def stanford_input_split_sentences():
    """
    Convert sentences into Stanford NLP input files
    Want to use the sentences generated by the custom python splitter so separate them by a newline
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/sentences_POS.csv'))

    # need to create list of files to be processed
    file_list = 'reuters/stanford_input/file_list.txt'
    file_list = os.path.abspath(os.path.join(basepath, '..', file_list))

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        with open(file_list, 'w') as f_list:
            src = 0

            for row in csv_reader:

                if row['SOURCE_ID'] != src:
                    # create new file for each record
                    file_out = 'reuters/stanford_input/' + row['SOURCE_ID'] + '.txt'
                    file_out = os.path.abspath(os.path.join(basepath, '..', file_out))
                    # write filepath to list
                    f_list.write(file_out + '\n')
                    src = row['SOURCE_ID']

                with open(file_out, 'a') as f_out:
                    f_out.write(row['SENTENCE'] + '\n')


def stanford_input():
    """
    Convert text into Stanford NLP input files
    This means that you need to use the stanford sentence splitter though :(
    """
    # TODO possibly add something to run the bash script after this is done
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/single_records.csv'))

    # need to create list of files to be processed
    file_list = 'reuters/stanford_input/file_list.txt'
    file_list = os.path.abspath(os.path.join(basepath, '..', file_list))

    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        with open(file_list, 'w') as f_list:

            for row in csv_reader:
                # create new file for each record
                file_out = 'reuters/stanford_input/' + row['SOURCE_ID'] + '.txt'
                file_out = os.path.abspath(os.path.join(basepath, '..', file_out))
                # write filepath to list
                f_list.write(file_out + '\n')

                with open(file_out, 'w') as f_out:
                    f_out.write(row['FRAGMENT'])

if __name__ == '__main__':
    stanford_input()