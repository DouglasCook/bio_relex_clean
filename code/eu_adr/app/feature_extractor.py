import re
import operator
import nltk
import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer


class FeatureExtractor():
    """
    Class for extracting features from previously tagged and chunked sets of words
    """
    # stopwords for use in cleaning up sentences
    stopwords = nltk.corpus.stopwords.words('english')

    def __init__(self, word_features=False, word_gap=True, count_dict=True, phrase_count=True, combo=True, pos=True,
                 entity_type=True, bag_of_words=False, bigrams=False, before=True, between=True, after=True):
        """
        Store variables for which features to use
        """
        # types of features to be extracted
        self.word_features = word_features
        self.bag_of_words = bag_of_words
        self.bigrams = bigrams
        self.word_gap = word_gap
        self.count_dict = count_dict
        self.phrase_count = phrase_count
        self.pos = pos
        self.combo = combo
        self.entity_type = entity_type
        self.before = before
        self.between = between
        self.after = after

        # create dicts for word features if they are going to be used
        if word_features:
            self.bef_verb_dict, self.bet_verb_dict, self.aft_verb_dict = {}, {}, {}
            self.bef_noun_dict, self.bet_noun_dict, self.aft_noun_dict = {}, {}, {}

    def generate_features(self, records, no_class=False, balance_classes=False):
        """
        Generate feature vectors and class labels for given records
        If no_class is true then the relation has not been annotated so don't return a class vector
        """
        feature_vectors = []
        class_vector = []

        for row in records:
            # store class label
            if not no_class:
                if row['true_rel']:
                    class_vector.append(1)
                else:
                    class_vector.append(0)

            f_dict = {}
            if self.entity_type:
                # add type of each entity
                f_dict.update({'type1': row['type1'], 'type2': row['type2']})
            # now add the features for each part of text
            if self.before:
                f_dict.update(self.part_feature_vectors(eval(row['before_tags']), 'before'))
            if self.between:
                f_dict.update(self.part_feature_vectors(eval(row['between_tags']), 'between'))
            if self.after:
                f_dict.update(self.part_feature_vectors(eval(row['after_tags']), 'after'))

            # now add whole dictionary to list
            feature_vectors.append(f_dict)

        if no_class:
            return feature_vectors
        # if a class is to be undersampled
        elif balance_classes:
            feature_vectors, class_vector = self.balance_classes(feature_vectors, class_vector)

        # return data as numpy array for easier processing
        return np.array(feature_vectors), np.array(class_vector)

    def part_feature_vectors(self, tags, which_set):
        """
        Generate features for a set of words, before, between or after
        """
        f_dict = {}

        # CHUNKS - only consider beginning tags of phrases, needs to be done before stopwords are removed
        phrases = [t[2] for t in tags if t[2] and t[2][0] == 'B']
        if self.phrase_count:
            # count number of each type of phrase
            f_dict['nps'] = sum(1 for p in phrases if p == 'B-NP')
            f_dict['vps'] = sum(1 for p in phrases if p == 'B-VP')
            # don't include prepositional phrases
            #f_dict['pps'] = sum(1 for p in phrases if p == 'B-PP')

        # remove stopwords and things not in chunks
        tags = [t for t in tags if t[0] not in self.stopwords and t[2] != 'O']

        # count word gap between entities only
        if self.word_gap and which_set == 'between':
            f_dict[which_set] = len(tags)

        # this uses a bag of words representation, no other features
        if self.bag_of_words:
            # ignore numbers
            words = [t[0] for t in tags if not re.match('.?\d', t[0])]
            for w in words:
                f_dict[w] = 1
            if self.bigrams:
                bigrams = ['-'.join([w[0], w[1]]) for w in zip(words, words[1:])]
                for b in bigrams:
                    f_dict[b] = 1

        if self.word_features:
            # don't take numbers into consideration, is this justifiable?
            # WORDS - check for presence of particular words
            verbs = [t[0] for t in tags if t[1][0] == 'V']
            nouns = [t[0] for t in tags if t[1][0] == 'N']
            f_dict.update(self.word_check(verbs, nouns, which_set))

        if self.pos:
            # POS - remove NONE tags here, seems to improve results slightly, shouldn't use untaggable stuff
            pos = [t[1] for t in tags if t[1] != '-NONE-']
            # if adjectives and adverbs are to be ignored
            #pos = [t[1] for t in tags if t[1] != '-NONE-' and t[1][0] not in ['J', 'R']]
            # use counting or non counting based on input parameter
            if self.count_dict:
                f_dict.update(self.counting_dict(pos))
            else:
                f_dict.update(self.non_counting_dict(pos))

        # COMBO - combination of tag and phrase type
        if self.combo:
            # slice here to remove 'B-'
            combo = ['-'.join([t[1], t[2][2:]]) for t in tags if t[2] and t[1] != '--NONE--']
            # use counting or non counting based on input parameter
            if self.count_dict:
                f_dict.update(self.counting_dict(combo))
            else:
                f_dict.update(self.non_counting_dict(combo))

        return f_dict

    def generate_word_features(self, records, data):
        """
        Generate the word features for given records and update feature dicts in original data
        This is done separately so it can be called once per split in learning curve creation
        """
        for i, row in enumerate(records):
            # generate features for each part of sentence and add to dict
            word_f_dict = self.one_set_word_features(eval(row['before_tags']), 'before')
            word_f_dict.update(self.one_set_word_features(eval(row['between_tags']), 'between'))
            word_f_dict.update(self.one_set_word_features(eval(row['after_tags']), 'after'))

            # now add word feature dict to feature dict for this relation
            data[i].update(word_f_dict)

    def one_set_word_features(self, tags, which_set):
        """
        Generate set of word features for one part of sentence
        """
        verbs = [t[0] for t in tags if t[1][0] == 'V']
        nouns = [t[0] for t in tags if t[1][0] == 'N']
        return self.word_check(verbs, nouns, which_set)

    def create_dictionaries(self, records, how_many):
        """
        Create dictionaries for most common verbs and nouns occurring in each part of the sentence
        how_many is the number of words to use unless = -1 in which case use all words
        """
        bef_verbs, bef_nouns, bet_verbs, bet_nouns, aft_verbs, aft_nouns = [], [], [], [], [], []

        # first create dictionaries of all terms
        # this makes counting terms much faster since don't need to check if key exists
        # words are stemmed before being written to database so no need to stem here
        for row in records:
            before = [t for t in eval(row['before_tags']) if t[0] not in self.stopwords]
            bef_verbs.extend([t[0] for t in before if t[1][0] == 'V'])
            bef_nouns.extend([t[0] for t in before if t[1][0] == 'N'])

            between = [t for t in eval(row['between_tags']) if t[0] not in self.stopwords]
            bet_verbs.extend([t[0] for t in between if t[1][0] == 'V'])
            bet_nouns.extend([t[0] for t in between if t[1][0] == 'N'])

            after = [t for t in eval(row['after_tags']) if t[0] not in self.stopwords]
            aft_verbs.extend([t[0] for t in after if t[1][0] == 'V'])
            aft_nouns.extend([t[0] for t in after if t[1][0] == 'N'])

        bef_verb_dict = {v: 0 for v in set(bef_verbs)}
        bef_noun_dict = {n: 0 for n in set(bef_nouns)}

        bet_verb_dict = {v: 0 for v in set(bet_verbs)}
        bet_noun_dict = {n: 0 for n in set(bet_nouns)}

        aft_verb_dict = {v: 0 for v in set(aft_verbs)}
        aft_noun_dict = {n: 0 for n in set(aft_nouns)}

        # now loop through and count occurrences
        for row in records:
            before = [t for t in eval(row['before_tags']) if t[0] not in self.stopwords]
            for t in before:
                if t[1][0] == 'V':
                    bef_verb_dict[t[0]] += 1
                elif t[1][0] == 'N':
                    bef_noun_dict[t[0]] += 1

            between = [t for t in eval(row['between_tags']) if t[0] not in self.stopwords]
            for t in between:
                if t[1][0] == 'V':
                    bet_verb_dict[t[0]] += 1
                elif t[1][0] == 'N':
                    bet_noun_dict[t[0]] += 1

            after = [t for t in eval(row['after_tags']) if t[0] not in self.stopwords]
            for t in after:
                if t[1][0] == 'V':
                    aft_verb_dict[t[0]] += 1
                elif t[1][0] == 'N':
                    aft_noun_dict[t[0]] += 1

        # if all words are to be used then slice is whole list
        if how_many == -1:
            how_many = None

        # now order by occurrence descending and store
        bef_verb_dict = sorted(bef_verb_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        bef_noun_dict = sorted(bef_noun_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        self.bef_verb_dict = [x[0] for x in bef_verb_dict[:how_many]]
        self.bef_noun_dict = [x[0] for x in bef_noun_dict[:how_many]]

        bet_verb_dict = sorted(bet_verb_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        bet_noun_dict = sorted(bet_noun_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        self.bet_verb_dict = [x[0] for x in bet_verb_dict[:how_many]]
        self.bet_noun_dict = [x[0] for x in bet_noun_dict[:how_many]]

        aft_verb_dict = sorted(aft_verb_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        aft_noun_dict = sorted(aft_noun_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
        self.aft_verb_dict = [x[0] for x in aft_verb_dict[:how_many]]
        self.aft_noun_dict = [x[0] for x in aft_noun_dict[:how_many]]

    def word_check(self, verbs, nouns, which_set):
        """
        Create features for most commonly occurring words in each part of sentence
        Words are already stemmed so list contains stemmed versions
        """
        if which_set == 'before':
            verb_list = self.bef_verb_dict
            noun_list = self.bef_noun_dict
        elif which_set == 'between':
            verb_list = self.bet_verb_dict
            noun_list = self.bet_noun_dict
        else:
            verb_list = self.aft_verb_dict
            noun_list = self.aft_noun_dict

        # if item doesn't exist in dict scikit vectoriser will default it to false
        stem_dict = {}
        for v in verb_list:
            # if the stem is found in the words set to true in dictionary
            if v in verbs:
                # need to make sure the same stem in different part of sentence doesn't override existing value
                stem_dict[which_set + '_v_' + v] = 1
        for n in noun_list:
            if n in nouns:
                stem_dict[which_set + '_n_' + n] = 1

        return stem_dict

    @staticmethod
    def balance_classes(feature_vectors, class_vector):
        """
        Undersample the over-represented class so it contains same number of samples
        """
        true_count = sum(class_vector)
        false_count = len(class_vector) - true_count

        together = sorted(zip(class_vector, feature_vectors))
        # split into classes
        false = together[:false_count]
        true = together[false_count+1:]

        # use seed so experiment is repeatable
        random.seed(0)

        # undersample the over represented class
        if true_count < false_count:
            false = random.sample(false, true_count)
        elif false_count < true_count:
            true = random.sample(true, false_count)

        # put back together again and shuffle so the classes are not ordered
        print len(true), len(false)
        together = false + true
        random.shuffle(together)

        # unzip before returning
        classes, features = zip(*together)

        return features, classes

    @staticmethod
    def counting_dict(tags):
        """
        Record counts of each tag present in tags and return as dictionary
        """
        c_dict = {}
        for t in tags:
            # if key exists increment count otherwise add the key
            if t in c_dict.keys():
                c_dict[t] += 1
            else:
                c_dict[t] = 1
        return c_dict

    @staticmethod
    def non_counting_dict(tags):
        """
        Record counts of each tag present in tags and return as dictionary
        """
        c_dict = {}
        for t in tags:
            # if key exists add it to dict
            if t not in c_dict.keys():
                c_dict[t] = 1
        return c_dict


if __name__ == '__main__':
    pass
