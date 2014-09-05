import nltk
import os
import csv
import re


class BigramChunker(nltk.ChunkParserI):
    """
    Bigram based chunker
    Should give reasonable results and require much less work than regex based one
    Actually doesn't work particularly well... maybe should use Stanford instead?
    """

    def __init__(self, train_sents):
        """
        The constructor takes a training data set and trains the classifier
        """
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        """
        Takes POS tagged sentence and returns a chunk tree
        """
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        # TODO check if this should or should not be done???
        # want to strip leading and trailing punctuation here so the chunk tags better match the pos tags
        conlltags = [(word.strip('"\'-.,'), pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        #return nltk.chunk.util.conlltags2tree(conlltags)
        return conlltags


def set_up_chunker():
    """
    Return trained chunker
    """
    # other option is the treebank chunk corpus
    train_sents = nltk.corpus.conll2000.chunked_sents('train.txt')
    return BigramChunker(train_sents)


def remove_punct():
    """
    Remove punctuation from chunked tags
    """
    # set filepath to input
    basepath = os.path.dirname(__file__)
    file_in = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/sentences_POS.csv'))
    file_out = os.path.abspath(os.path.join(basepath, '..', 'reuters/csv/clean_chunks.csv'))

    with open(file_in, 'rb') as csv_in:
        with open(file_out, 'wb') as csv_out:
            csv_reader = csv.DictReader(csv_in, delimiter=',')
            csv_writer = csv.DictWriter(csv_out, ['SOURCE_ID', 'SENT_NUM', 'POS_TAGS', 'CHUNKS'], delimiter=',')
            csv_writer.writeheader()

            for row in csv_reader:
                chunks = eval(row['CHUNKS'])
                tags = eval(row['POS_TAGS'])
                chunks = [(word, tag, chunk) for (word, tag, chunk) in chunks
                          if re.match(r'[\w\$\xc2_]+', word) is not None or chunk != 'O']

                if len(chunks) != len(tags):
                    print len(chunks), 'chunks and', len(tags), 'tags'
                    print chunks
                    print tags

    print 'Written to entities_marked.csv'


def test():
    # can use either of the corpora below for training
    train_sents = nltk.corpus.conll2000.chunked_sents('train.txt')
    test_sents = nltk.corpus.conll2000.chunked_sents('test.txt')
    #train_sents = nltk.corpus.treebank_chunk.chunked_sents()[:3000]
    #test_sents = nltk.corpus.treebank_chunk.chunked_sents()[3000:]
    chunker = BigramChunker(train_sents)
    test_s = 'In return, Bristol-Myers Squibb will obtain worldwide licensing rights to the recently discovered antiangiogenic applications of the drugs thalidomide, thalidomide analogs and Angiostatin (TM) protein, which have generated considerable interest in the medical community.'
    test_s = nltk.word_tokenize(test_s)
    test_s = nltk.pos_tag(test_s)
    tree = chunker.parse(test_s)
    print tree
    #print chunker.evaluate(test_sents)


if __name__ == '__main__':
    test()
