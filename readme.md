# README

## Set up Python and the required modules

Various modules are used for the calculations, retrieval of additional abstracts and the web app 

If GCC is not already present on the machine it must be installed first

* `sudo apt-get GCC`

pip is a package installer for python which makes the rest of the installation steps easier

* `sudo apt-get python-pip`

The following are modules required for the computations, and some dependencies they have

* `sudo apt-get install build-essential python-dev`
* `sudo apt-get build-dep python-numpy python-scipy`
* `sudo pip install -U numpy`
* `sudo pip install -U scipy`
* `sudo pip install -U scikit-learn`

Now NLTK for the natural language processing

* `sudo pip install -U nltk`

Also need to install the corpora if you are using any,  in our case we want the conll2000 corpus to use for the chunker.

This will install all of the NLTK corpora, including conll2000  * `python -m nltk.downloader -d /usr/share/nltk_data all`  

biopython is for interfacing with the Entrez database engine to download PubMed abstracts

* `sudo pip install -U biopython`
* `sudo pip install -U Unidecode`

Now for the server and web app modules

* `sudo pip install -U pyyaml`
* `sudo pip install -U flask`

## Creating the starting database

The data for the original database comes from the EU-ADR and BioText corpora discussed in the report, each is preprocessed differently

BioText preprocessing is carried out using `code/biotext/biotext_preprocessing.py`

For EU-ADR the abstracts are fetched using `code/eu_adr/get_abstracts.py` and preprocessed using `code/eu_adr/preprocess.py`

The database management system used is SQLite, if not already present it can be installed with

* `sudo apt-get install sqlite3 libsqlite3-dev`

The `code/eu_adr/database` directory contains both the starting corpus `euadr_biotext.db` and the final corpus containing the newly annotated relations `relex.db`

The starting corpus can be created using the `intial_setup` function in `db_creation.py`, assuming the correct csv file is present

## Retrieving additional texts from PubMed

`code/eu_adr/get_abstracts.py` contains functions for this

Firstly an Entrez query must be run using `pubmed_query()` or a modified version

Then the actual abstracts are pulled down from pubmed with `retrieve_abstracts()`

These can then be entered into the database using `medline_to_db()`

Preprocessing should then be carried out using the functions in `code/eu_adr/pubmed_preprocess.py`

## Running the server

To enable the annotation of the newly entered records the web app can be used, all functionality is contained in the `code/eu_adr/app` directory

To run the server locally:

`python server.py 0`

Or to run it on the current IP and port specified in server.py (currently 55100):

`python server.py 1`

## Analysing the data

Everything for this is contained in the `code/eu_adr` directory
Grid search for parameter tuning and kernel selection can be carried out with `analysis_new.py`

Learning curves can be generated with `new_data_curves.py` and `cross_validation_curves.py` and plotted nicely using `draw_nice_plots.py`

Everything above will need to be tweaked depending on the tests to be performed.
