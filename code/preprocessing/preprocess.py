import nltk
import nltk.tokenize.punkt as punkt

import csv  # used for accessing data held in CSV format
import os.path  # need this to use relative filepaths
import sys  # using sys for std out
import re

import chunking


def set_up_tokenizer():
    """
    Set up sentence splitter with custom parameters and return to caller
    """
    punkt_params = punkt.PunktParameters()
    # sentences are not split ending on the given parameters, using {} creates a set literal
    punkt_params.abbrev_types = {'inc', '.tm', 'tm', 'no', 'i.v', 'dr', 'drs', 'u.s', 'u.k', 'ltd', 'vs', 'vol', 'corp',
                                 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
                                 'pm', 'p.m', 'am', 'a.m', 'mr', 'mrs', 'ms', 'i.e'}
    # the tokenizer has to be unpickled so better do it once here than every time it is used
    return punkt.PunktSentenceTokenizer(punkt_params)


def remove_punctuation(sentence):
    """
    Remove all punctuation from sentence
    Return original sentence and that with no punctuation
    """
    # want to keep hyphenated words but none of the other hyphens
    # replace any hyphenated words' hyphens with underscores
    for hyphenated in re.findall(r'\w-\w', sentence):
        underscored = hyphenated.replace('-', '_')
        sentence = sentence.replace(hyphenated, underscored)

    # TODO this is removing decimal points and percentages, need to fix it
    # %% is how to escape a percentage in python regex

    # remove punctuation, still want to add original sentence to CSV though
    #no_punct = re.findall(r'[\w\$\xc2()-]+', s)
    no_punct = re.findall(r'[\w\$\xc2_]+', sentence)
    no_punct = ' '.join(no_punct)

    # may not want to do this, underscores work nicer when chunking
    # put the hyphens back
    #sentence = sentence.replace('_', '-')

    return sentence, no_punct


def contains_both(text, drugs, companies):
    """
    Check whether the text contains at least one of the drugs and one of the companies
    """
    # loop through drugs and companies to see if one of each is mentioned
    company_suffixes = ['inc', 'gmbh', 'co', 'plc', 'group', 'corp', 'ltd', 'ag', 'a/s']
    for d in drugs:
        if d in text:
            for c in companies:
                # if the company has a common suffix check for the leading word only
                tokens = c.split()
                if tokens[-1].lower() in company_suffixes:
                    c = tokens[0]
                if c in text:
                    return True

    return False


def collate_texts(delimiter='\t'):
    """
    Input spreadsheet contains one row per drug-company pair, so the same text appears on multiple rows
    Create one record per text fragment, with lists for all drugs and companies
    Only keep those texts that mention at least one of the drugs and one of the companies
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    #file_in = 'data/reuters/press_releases/PR_drug_company_500.csv'
    file_in = 'data/reuters/TR_PR_DrugCompany.csv'
    file_in = os.path.abspath(os.path.join(basepath, '..', '..', file_in))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/single_records.csv'))

    #with open(file_in, 'rb') as csv_in:
    # may need to open with rU to deal with universal newlines - something to do with excel
    with open(file_in, 'rU') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=delimiter)
            csv_writer = csv.writer(csv_out, delimiter=',')

            drugs = set([])
            companies = set([])
            src = '0'
            text = ''

            csv_writer.writerow(['SOURCE_ID', 'DRUGS', 'COMPANIES', 'FRAGMENT'])

            # think that the dict reader skips header row automagically
            for row in csv_reader:

                # src != 0 so the first row will always be included
                if row['SOURCE_ID'] != src and src != '0':
                    # first check if the text contains at least on of each drug and company tagged
                    if contains_both(text, drugs, companies):
                        csv_writer.writerow([src, list(drugs), list(companies), nltk.clean_html(text)])

                    # reset lists
                    drugs = set([])
                    companies = set([])

                # append drug and company to lists
                drugs.add(row['DRUG_NAME'])
                companies.add((row['COMPANY_NAME']))
                src = row['SOURCE_ID']
                text = row['FRAGMENT']

    print 'Written to single_records.csv'


def clean_and_tag_all():
    """
    Create new CSV containing tagged versions of all sentences
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/single_records.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/sentences_POS.csv'))

    sentence_splitter = set_up_tokenizer()
    chunker = chunking.set_up_chunker()
    stemmer = nltk.SnowballStemmer('english')

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, ['SOURCE_ID', 'DRUGS', 'COMPANIES', 'SENTENCE'], delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['SOURCE_ID', 'SENT_NUM', 'SENTENCE', 'NO_PUNCT',
                                                  'DRUGS', 'COMPANIES', 'POS_TAGS', 'CHUNKS'], delimiter=',')
            csv_writer.writeheader()
            #csv_reader.next()

            for row in csv_reader:
                # display progress bar
                sys.stdout.write('.')
                sys.stdout.flush()

                # clean up html tags
                # named SENTENCE in the reader so it works nicely when writing row
                plaintext = nltk.clean_html(row['SENTENCE'])
                # this in particular seems to be screwing up some of the sentence splitting
                plaintext = plaintext.replace('Inc .', 'Inc.')
                # split into sentences
                sentences = sentence_splitter.tokenize(plaintext)

                if len(sentences) > 0:
                    for i, s in enumerate(sentences):

                        # TODO integrate stanford NER recognition output into this

                        # clean up sentence
                        s, no_punct = remove_punctuation(s)

                        # CHUNKING - need to include punctuation for this to be anywhere near accurate
                        tokens = nltk.pos_tag(nltk.word_tokenize(s))
                        chunks = chunker.parse(tokens)

                        # POS TAGS - don't want to include punctuation
                        tokens = nltk.word_tokenize(no_punct)
                        # put the hyphens back after tokenisation
                        # underscores mean that the tokens are better recognised when tagging
                        no_punct = no_punct.replace('_', '-')
                        s = s.replace('_', '-')
                        tags = nltk.pos_tag(tokens)

                        # STEMMING - add stemmed version of word to end of each tagged token
                        tags = [(token, tag, stemmer.stem(token.lower())) for (token, tag) in tags]

                        # TODO parse tree info, chunking, something to do with stemming?
                        # ignore any rogue bits of punctuation etc
                        if len(tags) > 1:
                            # write row to file for each sentence
                            new_fields = {'SENT_NUM': i, 'SENTENCE': s, 'NO_PUNCT': no_punct, 'POS_TAGS': tags,
                                          'CHUNKS': chunks}
                            row.update(new_fields)
                            csv_writer.writerow(row)

    print 'Written to sentences_POS.csv'


def preprocessing():
    """
    Step 1 in the pipeline so far...
    Retrieve and clean relevant texts from CSV and carry out POS tagging
    """
    collate_texts('\t')
    clean_and_tag_all()


if __name__ == '__main__':
    collate_texts('\t')
    #preprocessing()
    #stanford_entity_recognition()
