import os
import csv


def count_sentences_boo():
    """ Count number of relevant sentences"""

    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = 'data/reuters/press_releases/PR_drug_company_500.csv'
    file_in = os.path.abspath(os.path.join(basepath, '..', '..', file_in))

    count = 0
    with open(file_in, 'rb') as csv_in:
        csv_reader = csv.DictReader(csv_in, delimiter=',')

        for row in csv_reader:
            if row['DRUG_NAME'] in row['FRAGMENT'] and row['COMPANY_NAME'] in row['FRAGMENT']:
                count += 1

        print count
