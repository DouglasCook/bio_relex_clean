import nltk
import csv              # used for accessing data held in CSV format
import os.path          # need this to use relative filepaths
import sys              # using sys for std out
import re

import preprocessing

import nltk.tokenize.punkt as punkt


def clean_and_tag():
    """ Create new CSV containing all relevant sentences """

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = 'data/reuters/press_releases/PR_drug_company_500.csv'
    file_in = os.path.abspath(os.path.join(basepath, '..', '..', file_in))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/sentences_POS.csv'))

    # set up sentence splitter with custom parameters
    punkt_params = punkt.PunktParameters()
    # sentences are not split ending on the given parameters, using {} creates a set literal
    punkt_params.abbrev_types = {'inc', 'inc ', '.tm', 'tm', 'no', 'i.v', 'drs', 'u.s'}
    # the tokenizer has to be unpickled so better do it once here than every time it is used
    sentence_splitter = punkt.PunktSentenceTokenizer(punkt_params)

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            # TO DO use dictionary reader to avoid using magic numbers for columns
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            # write column headers on first row
            row = csv_reader.next()
            row.append('POS TAGS')
            csv_writer.writerow(row)

            for row in csv_reader:
                # use stdout to avoid spaces and newlines
                sys.stdout.write('.')
                # need to flush the buffer to display immediately
                sys.stdout.flush()

                # clean up html tags
                plaintext = nltk.clean_html(row[1])
                drug = row[3]
                company = row[5]
                src = row[0]

                # only consider texts containing both the drug and company
                if drug in plaintext and company in plaintext:
                    sentences = sentence_splitter.tokenize(plaintext)

                    # filter for only sentences mentioning drug, company or both
                    # TO DO coreference resolution to find more relevant sentences
                    sentences = [s for s in sentences if drug in s or company in s]

                    if len(sentences) > 0:
                        for s in sentences:
                            # remove punctuation, still want to add original sentence to CSV though
                            no_punct = re.findall(r'[\w\$\xc2()-]+', s)
                            no_punct = ' '.join(no_punct)
                            tokens = nltk.word_tokenize(no_punct)
                            tags = nltk.pos_tag(tokens)

                            # TO DO parse tree info, something to do with stemming?
                            # write row to file for each sentence
                            row.append(tags)
                            csv_writer.writerow([src, s, row[2], drug, row[4], company, tags])


def only_easy():
    """
    Strip out any sentences not mentioning both drug and company
    """
    with open('data/sentences_chunk.csv', 'rb') as csv_in:
        with open('data/sentences_easy_set.csv', 'wb') as csv_out:
            csv_reader = csv.reader(csv_in, delimiter=',')
            csv_writer = csv.writer(csv_out, delimiter=',')

            row = csv_reader.next()
            csv_writer.writerow(row)

            for row in csv_reader:
                if row[3] in row[1] and row[5] in row[1]:
                    csv_writer.writerow(row)


if __name__ == '__main__':
    clean_and_tag()
    preprocessing.chunk('reuters/sentences_POS.csv', 'reuters/sentences_chunk.csv')
