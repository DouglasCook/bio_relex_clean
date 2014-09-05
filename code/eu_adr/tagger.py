import re
import nltk


class TaggerChunker(nltk.ChunkParserI):
    """
    Bigram based chunker
    Should give reasonable results and require much less work than regex based one
    Actually doesn't work particularly well... maybe should use Stanford instead?
    """
    def __init__(self):
        """
        The constructor takes a training data set and trains the classifier
        """
        # conll2000 chunker is more detailed than the treebank one ie includes prepositional chunks
        train_sents = nltk.corpus.conll2000.chunked_sents('train.txt')
        #train_sents = nltk.corpus.treebank_chunk.chunked_sents()
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]

        self.tagger = nltk.BigramTagger(train_data)
        self.stemmer = nltk.SnowballStemmer('english')

    def parse(self, sentence):
        """
        Takes POS tagged sentence and returns a chunk tree
        """
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag) in zip(sentence, chunktags)]
        return conlltags

    def pos_and_chunk_tags(self, sentence, before, between, e1, e2):
        """
        Return word, pos tag, chunk triples
        """
        sentence = nltk.word_tokenize(sentence)

        # now calculate start and end token indices
        bef_end = len(nltk.word_tokenize(before))
        bet_start = bef_end + len(nltk.word_tokenize(e1))
        bet_end = bet_start + len(nltk.word_tokenize(between))
        aft_start = bet_end + len(nltk.word_tokenize(e2))

        # text = [b for b in between if b not in stopwords]
        tags = nltk.pos_tag(sentence)
        all_chunks = self.parse(tags)

        # now split sentence into its parts and tidy up
        bef_chunks = self.clean_chunks(all_chunks[:bef_end])
        bet_chunks = self.clean_chunks(all_chunks[bet_start:bet_end])
        aft_chunks = self.clean_chunks(all_chunks[aft_start:])

        return bef_chunks, bet_chunks, aft_chunks

    def clean_chunks(self, chunks):
        """
        Stem the words and remove punctuation
        """
        # now want to remove any punctuation - maybe don't want to remove absolutely all punctuation?
        chunks = [c for c in chunks if not re.match('\W', c[0])]

        # stemming - not sure about the encode decode nonsense...
        chunks = [(self.stemmer.stem(c[0].decode('utf-8')), c[1], c[2]) for c in chunks]
        chunks = [(c[0].encode('utf-8'), c[1], c[2]) for c in chunks]

        return chunks


if __name__ == '__main__':
    chunker = TaggerChunker()
    stopwords = nltk.corpus.stopwords.words('english')
    sent = 'Sucroferric oxyhydroxide was launched in the USA in 2014 for the treatment of hyperphosphatemia in adult dialysis patients.'
    tokens = nltk.word_tokenize(sent)
    important = [t for t in tokens if t not in stopwords]
    print important
    tags = nltk.pos_tag(tokens)
    print tags
    chunks = chunker.parse(tags)
    print chunks
