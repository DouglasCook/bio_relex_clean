# README

Many things need to be installed before you can run anything!

## Set up Python

Only need GCC if it't not on the VM (for whatever bizarre reason...)

* `sudo apt-get GCC`
* `sudo apt-get python-pip`
* `sudo apt-get install build-essential python-dev`
* `sudo apt-get build-dep python-numpy python-scipy`
* `sudo pip install -U numpy`
* `sudo pip install -U scipy`
* `sudo pip install -U scikit-learn`
* `sudo pip install -U biopython`
* `sudo pip install -U Unidecode`
* `sudo pip install -U pyyaml`

## Set up NLTK
* sudo pip install -U nltk

Also need to install the corpora if you are using any, 
in our case we want the conll2000 to use for the chunker.

This will install all the NLTK corpora so possibly overkill!

* `python -m nltk.downloader -d /usr/share/nltk_data all`

## Stanford NER tool

Need to install Java!

* `wget http://nlp.stanford.edu/software/stanford-ner-2014-06-16.zip`
