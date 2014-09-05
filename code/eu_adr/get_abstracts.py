import os
import urllib2
import pickle
import csv
import xml.etree.cElementTree as ET  # python XML manipulation library, C version because it's way faster!
import sqlite3

import nltk.tokenize.punkt as punkt

from Bio import Entrez
from Bio import Medline
Entrez.email = 'dsc313@imperial.ac.uk'
db_path = 'database/test.db'


def scrape_pubmed():
    """
    Download XML version of abstracts used in EU-ADR data set
    """
    basepath = os.path.dirname(__file__)
    files = os.listdir('/Users/Dug/Imperial/individual_project/data/euadr_corpus')

    for f in files:
        pubmed_id = f.split('.')[0]
        # avoid random crap files eg ds_store
        if len(pubmed_id) > 0:
            # want xml version for easier processing
            url = 'http://www.ncbi.nlm.nih.gov/pubmed/?term=' + pubmed_id + '&report=xml&format=text'
            fp = os.path.abspath(os.path.join(basepath, 'abstracts', pubmed_id + '.xml'))

            raw = urllib2.urlopen(url).read()
            # replace HTML literals
            raw = raw.replace('&lt;', '<')
            raw = raw.replace('&gt;', '>')

            with open(fp, 'wb') as f_out:
                f_out.write(raw)


def file_list():
    """
    Pickle a list containing all pubmed IDs corresponding to EU-ADR corpus 
    """
    files = os.listdir('/Users/Dug/Imperial/individual_project/data/euadr_corpus')
    id_list = []
    for f in files:
        parts = f.split('.')
        if parts[1] == 'csv':
            id_list.append(parts[0])

    # now pickle it for later use
    pickle.dump(id_list, open('eu-adr_ids.p', 'wb'))


def set_up_tokenizer():
    """
    Set up sentence splitter with custom parameters and return to caller
    """
    punkt_params = punkt.PunktParameters()
    # sentences are not split ending on the given parameters, using {} creates a set literal
    punkt_params.abbrev_types = {'inc', '.tm', 'tm', 'no', 'i.v', 'dr', 'drs', 'u.s', 'u.k', 'ltd', 'vs', 'vol', 'corp',
                                 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
                                 'pm', 'p.m', 'am', 'a.m', 'mr', 'mrs', 'ms', 'i.e', 'e.g',
                                 # above is from reuters, below for eu-adr specifically
                                 'spp'}

    return punkt.PunktSentenceTokenizer(punkt_params)


def fix_html(text):
    """
    Sort out HTML nonsense
    """
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&amp;', '&')
    return text


def xml_to_csv():
    """
    Create one record per text using XML abstracts scraped from PubMed
    """
    basepath = os.path.dirname(__file__)
    # unpickle list of eu-adr ids
    files = pickle.load(open('eu-adr_ids.p', 'rb'))
    files = [os.path.abspath(os.path.join(basepath, 'abstracts', f + '.xml')) for f in files]
    f_out = os.path.abspath(os.path.join(basepath, 'csv', 'sentences.csv'))

    sentence_splitter = set_up_tokenizer()

    with open(f_out, 'wb') as csv_out:
        csv_writer = csv.DictWriter(csv_out, ['id', 'sent_num', 'text'], delimiter=',')
        csv_writer.writeheader()

        for f in files:
            # parse the tree
            tree = ET.parse(f)

            # use xpath paths here to search entire tree
            pubmed_id = tree.findtext('.//PMID')
            title = tree.findtext('.//ArticleTitle')
            nodes = tree.findall('.//AbstractText')

            # problems arise here if the abstract is in several parts eg method, conclusion etc
            # so need to construct text in this longwinded way instead of nice comprehension above
            abstract = ''
            for n in nodes:
                # can check for empty list, dict or None by simply using if
                if n.attrib and n.attrib['Label'] and n.attrib['Label'] != 'UNLABELLED':
                    abstract += ' ' + n.attrib['Label'] + ': ' + n.text
                else:
                    abstract += ' ' + n.text
            text = title + abstract
            text = fix_html(text)

            sentences = sentence_splitter.tokenize(text)

            for i, s in enumerate(sentences):
                # dict comprehension here to hack the unicode into csv writer
                dict_row = {'id': pubmed_id, 'sent_num': i, 'text': s}
                csv_writer.writerow(dict_row)


def pubmed_query():
    """
    Perform search of Pubmed database using Entrez and return relevant ids
    """
    # query taken from eu-adr corpus with the adverse effects part removed
    query = ('("Inorganic Chemicals"[Mesh] OR "Organic Chemicals"[Mesh] OR "Heterocyclic Compounds"[Mesh] '
             'OR "Polycyclic Compounds"[Mesh] OR "Hormones, Hormone Substitutes, and Hormone Antagonists"[Mesh] '
             'OR "Carbohydrates"[Mesh] OR "Lipids"[Mesh] OR "Amino Acids, Peptides, and Proteins"[Mesh] '
             'OR "Nucleic Acids, Nucleotides, and Nucleosides"[Mesh] OR "Biological Factors"[Mesh] '
             'OR "Biomedical and Dental Materials"[Mesh] OR "Pharmacologic Actions"[Mesh]) '
             'AND "Diseases Category"[Mesh] AND hasabstract[text] AND ("2012"[PDAT]: "2013"[PDAT]) '
             'AND "humans"[MeSH Terms] AND English[lang]')
    # retmax defines how many ids to return, defaults to 20
    handle = Entrez.esearch(db='pubmed', term=query, retmax=1000)
    record = Entrez.read(handle)

    pickle.dump(record['IdList'], open('pickles/pubmed_records.p', 'wb'))


def pubmed_query_new():
    """
    Perform search of Pubmed database using Entrez and return relevant ids
    """
    # query specified by Andrew
    query = ('("crohn\'s disease"[All Fields] OR "chronic kidney disease"[All Fields]) '
             'AND hasabstract[text] AND English[lang] AND ("2012"[PDAT]: "2014"[PDAT])')
    # retmax defines how many ids to return, defaults to 20 so need to increase it
    handle = Entrez.esearch(db='pubmed', term=query, retmax=1000)
    record = Entrez.read(handle)

    pickle.dump(record['IdList'], open('pickles/pubmed_records_new.p', 'wb'))


def retrieve_abstracts():
    """
    Bring down abstracts from pubmed based on ids stored in pickle
    """
    record = pickle.load(open('pickles/pubmed_records_new.p', 'rb'))
    #record = pubmed_query_new()

    for pubmed_id in record:
        print pubmed_id
        # link to the file through Entrez - use medline format so it can be easily parsed by biopython
        handle = Entrez.efetch(db='pubmed', id=pubmed_id, rettype='medline', retmode='text')
        fp = 'pubmed/' + pubmed_id + '.txt'
        with open(fp, 'wb') as f_out:
            f_out.write(handle.read())


def medline_to_csv():
    """
    Create one record per text using XML abstracts scraped from PubMed
    """
    files = pickle.load(open('pickles/pubmed_records.p', 'rb'))
    files = ['pubmed/' + f + '.txt' for f in files]
    f_out = 'csv/sentences_pubmed.csv'

    sentence_splitter = set_up_tokenizer()

    with open(f_out, 'wb') as csv_out:
        csv_writer = csv.DictWriter(csv_out, ['id', 'sent_num', 'text'], delimiter=',')
        csv_writer.writeheader()

        for f in files:
            with open(f, 'rb') as f_in:
                record = Medline.read(f_in)
                # use medline parser to extract relevant data from the file
                pid = record['PMID']
                text = record['TI'] + ' ' + record['AB']

                sentences = sentence_splitter.tokenize(text)
                for i, s in enumerate(sentences):
                    # dict comprehension here to hack the unicode into csv writer
                    dict_row = {'id': pid, 'sent_num': i, 'text': s.encode('utf-8')}
                    csv_writer.writerow(dict_row)


def medline_to_db():
    """
    Create one record per text using medline abstracts scraped from PubMed
    """
    sentence_splitter = set_up_tokenizer()
    files = set(pickle.load(open('pickles/pubmed_records_new.p', 'rb')))

    with sqlite3.connect(db_path) as db:
        cursor = db.cursor()

        # don't want to add the same abstract multiple times, so get existing ones first
        cursor.execute('SELECT DISTINCT pubmed_id FROM sentences')
        # using sets to hopefully speed things up
        existing = {p[0] for p in cursor}
        files = {f for f in files if f not in existing}
        files = ['pubmed/' + str(f) + '.txt' for f in files]

        for f in files:
            with open(f, 'rb') as f_in:
                record = Medline.read(f_in)
                # use medline parser to extract relevant data from the file
                pid = record['PMID']

                try:
                    text = record['TI'] + ' ' + record['AB']
                # bti is for books? the value is a list for some reason so just take first element
                except:
                    text = record['BTI'][0] + ' ' + record['AB']

                sentences = sentence_splitter.tokenize(text)
                for i, s in enumerate(sentences):
                    cursor.execute('''INSERT INTO sentences
                                             VALUES (NULL, ?, ?, ?, ?);''', (pid, i, s, 'pubmed'))

if __name__ == '__main__':
    #pubmed_query()
    #scrape_pubmed()
    #file_list()
    pubmed_query_new()
    retrieve_abstracts()
    medline_to_db()
