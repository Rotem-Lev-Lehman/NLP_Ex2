import math
import re
from collections import Counter
from random import choices


class Spell_Checker:
    """The class implements a context sensitive spell checker. The corrections
        are done in the Noisy Channel framework, based on a language model and
        an error distribution model.
    """
    
    def __init__(self, lm=None):
        """Initializing a spell checker object with a language model as an
        instance  variable. The language model should suppport the evaluate()
        and the get_model() functions as defined in assignment #1.

        Args:
            lm: a language model object. Defaults to None
        """
        self.lm = lm
        self.error_tables = None
        self.denominator_error_tables = None
        self.vocabulary = None

    def build_model(self, text, n=3):
        """Returns a language model object built on the specified text. The language
            model should support evaluate() and the get_model() functions as defined
            in assignment #1.

            Args:
                text (str): the text to construct the model from.
                n (int): the order of the n-gram model (defaults to 3).

            Returns:
                A language model object
        """
        lm = Ngram_Language_Model(n=n, chars=False)
        lm.build_model(text)
        return lm

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM disctionary if set)

            Args:
                lm: a language model object
        """
        self.lm = lm
        self.update_vocabulary()
        self.update_error_tables_denominator()

    def update_vocabulary(self):
        """Updates the vocabulary to be according to the language model's "get_model" function.
            (Replaces an older vocabulary if set)
        """
        all_ngrams_dict = self.lm.get_model()
        self.vocabulary = set()
        for ngram in all_ngrams_dict.keys():
            split_words = ngram.split(' ')
            for word in split_words:
                self.vocabulary.add(word)

    def update_error_tables_denominator(self):
        """Updates the denominator of the error tables to be according to the language model's "get_model" function.
            (Replaces an older denominator table if set)
        """
        all_ngrams_dict = self.lm.get_model()
        # The denominator could be a combination of 1 or 2 letters count in the corpus.
        # Special case of #, it will be the amount of words,
        # (because it will mean 'how many times was there a start of a word?'):
        self.denominator_error_tables = {'one_letter': {'#': 0}, 'two_letters': {}}
        for ngram, count in all_ngrams_dict.items():
            split_words = ngram.split(' ')
            for word in split_words:
                # The first letter in the word is also counted, and we check how many times a word starts with that letter:
                prev_letter = '#'
                self.denominator_error_tables['one_letter'][prev_letter] += count  # count the amount of words.
                for letter in word:
                    if letter not in self.denominator_error_tables['one_letter'].keys():
                        self.denominator_error_tables['one_letter'][letter] = 0
                    self.denominator_error_tables['one_letter'][letter] += count

                    curr_entry = prev_letter + letter
                    # If we have not yet seen this entry before, add it to the dictionary:
                    if curr_entry not in self.denominator_error_tables['two_letters'].keys():
                        self.denominator_error_tables['two_letters'][curr_entry] = 0
                    # This entry occurred at least <count> times, because this entire ngram has occurred <count> times:
                    self.denominator_error_tables['two_letters'][curr_entry] += count
                    # Update the prev letter to be this letter:
                    prev_letter = letter

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {str: int} represents the confution matrix of the
            specific errors, where str is a string of two characters mattching the
            row and culumn "indixes" in the relevant confusion matrix and the int is the
            observed count of such an error (computed from the specified errors file).
            Examples of such string are 'xy', for deletion of a 'y'
            after an 'x', insertion of a 'y' after an 'x' and substitution
            of 'x' (incorrect) by a 'y'; and example of a transposition is 'xy' indicates the characters that are transposed.


            Notes:
                1. Ultimately, one can use only 'deletion' and 'insertion' and have
                    'substitution' and 'transposition' derived. Again,  we use all
                    four types explicitly in order to keep things simple.
            Args:
                errors_file (str): full path to the errors file. File format, TSV:
                                    <error>    <correct>


            Returns:
                A dictionary of confusion "matrices" by error type (dict).
        """
        error_tables = self.initialize_empty_error_tables()
        with open(errors_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            split_line = line.rstrip().lower().split('\t')
            error_word = split_line[0]
            correct_word = split_line[1]
            if len(error_word) < len(correct_word):
                # This can only be a result of deletion (xy typed as x):
                error_entry = self.find_deletion_error(error_word, correct_word)
                error_type = 'deletion'
            elif len(error_word) > len(correct_word):
                # This can only be a result of insertion (x typed as xy):
                # The insertion error can be calculated as a deletion error where the deleted letter is in the error_word:
                error_entry = self.find_deletion_error(correct_word, error_word)
                error_type = 'insertion'
            else:
                # This can be either a result of substitution (x typed as y) or transposition (xy typed as yx):
                error_type, error_entry = self.find_substitution_or_transposition(error_word, correct_word)
            if error_entry in error_tables[error_type].keys():  # only insert allowed entries ('xy' or '#x')
                error_tables[error_type][error_entry] += 1
        return error_tables

    def initialize_empty_error_tables(self):
        """ Returns a dictionary representing an empty error_table dictionary.

            Returns:
                dict. The dictionary is in the format returned by learn_error_tables,
                but with 0 as the count of each entry.
        """
        insertion_table = self.initialize_empty_confusion_matrix(with_start_word_letters=True)
        deletion_table = self.initialize_empty_confusion_matrix(with_start_word_letters=True)
        substitution_table = self.initialize_empty_confusion_matrix(with_start_word_letters=False)
        transposition_table = self.initialize_empty_confusion_matrix(with_start_word_letters=False)
        error_tables = {'insertion': insertion_table,
                        'deletion': deletion_table,
                        'substitution': substitution_table,
                        'transposition': transposition_table}

        return error_tables

    def initialize_empty_confusion_matrix(self, with_start_word_letters=True):
        """ Returns a dictionary representing an empty confusion matrix over all letters in the english language.

            Returns:
                dict. The confusion matrix in the format of: {str: int},
                where str is an entry 'xy', and the result int is 0.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        confusion_matrix = {}
        for l1 in letters:
            for l2 in letters:
                entry = l1 + l2
                confusion_matrix[entry] = 0
            if with_start_word_letters:
                entry_start_of_word = '#' + l1
                confusion_matrix[entry_start_of_word] = 0
        return confusion_matrix

    def find_deletion_error(self, short_word, long_word):
        """ Finds the deleted letter from the long_word.

            Note:
                len(short_word) must be less than len(long_word).
            Args:
                short_word (str): the shorter word which have a deleted letter.
                long_word (str): the longer word which have all of the letters.

            Returns:
                str. The error entry of the deletion.
        """
        deletion_index = None
        for i in range(len(short_word)):
            error_letter = short_word[i]
            correct_letter = long_word[i]
            if error_letter != correct_letter:
                # Found the deletion:
                deletion_index = i
                break
        if deletion_index is None:
            deletion_index = len(long_word) - 1  # if we haven't found the difference, it must be the last letter.
        if deletion_index == 0:
            error_entry = '#' + long_word[deletion_index]
        else:
            error_entry = long_word[deletion_index - 1:deletion_index + 1]
        return error_entry

    def find_substitution_or_transposition(self, error_word, correct_word):
        """ Finds the error in the given word, and determines if it is a substitution or a transposition error.

            Note:
                len(error_word) must be equal to len(correct_word).
            Args:
                error_word (str): the word that has the error in it.
                correct_word (str): the correct word.

            Returns:
                tuple. The tuple is in the form of (error_type, error_entry),
                where error_type is in ['substitution' / 'transposition'].
        """
        error_type = None
        error_entry = None
        for i in range(len(error_word)):
            if error_word[i] != correct_word[i]:
                # Found the error:
                if i + 1 < len(error_word) and error_word[i + 1] == correct_word[i] and error_word[i] == correct_word[i + 1]:
                    # Than it is a transposition error (xy typed as yx):
                    error_type = 'transposition'
                    error_entry = correct_word[i:i + 2]
                else:
                    # This must be a substitution error (x typed as y):
                    error_type = 'substitution'
                    error_entry = correct_word[i] + error_word[i]
                break
        return error_type, error_entry

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """
        self.error_tables = error_tables

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        return self.lm.evaluate(text)

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model if the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """
        nt = normalize_text(text)
        alpha_log = math.log(alpha)
        one_minus_alpha_log = math.log(1 - alpha)
        split_text = nt.split(' ')
        max_prob = None
        max_sentence = None
        for i, word in enumerate(split_text):
            all_edits = self.get_all_edits(word)
            for edited_word, prob in all_edits:
                sentence, sentence_prob = self.try_sentence_in_lm(split_text, i, edited_word, prob + one_minus_alpha_log)
                if max_prob is None or max_prob < sentence_prob:
                    max_prob = sentence_prob
                    max_sentence = sentence

            # This can be done once outside of the for loop, but it is done here to reflect the formula from the class:
            original_word_sentence, sentence_prob = self.try_sentence_in_lm(split_text, i, word, alpha_log)
            if max_prob is None or max_prob < sentence_prob:
                max_prob = sentence_prob
                max_sentence = original_word_sentence
        return max_sentence

    def try_sentence_in_lm(self, split_text, i, edited_word, edit_prob):
        """ Calculates the probability of the new sentence to occur in the language model,
            given the edited word and the edit's probability

            Args:
                split_text (list): the original text in a list form, so that it will be easy
                    to replace the original word with it's edited word.
                i (int): the index of the word we wish to replace.
                edited_word (str): the edited word we wish to use in the new sentence.
                edit_prob (float): the log-probability of the edit to occur.

            Return:
                tuple. The tuple shall be in the following form: (sentence, sentence_prob)
        """
        copy_of_list = []
        for j, word in enumerate(split_text):
            if i == j:
                copy_of_list.append(edited_word)
            else:
                copy_of_list.append(word)
        sentence = " ".join(copy_of_list)
        prob_sentence = self.evaluate(sentence)
        total_probability = prob_sentence + edit_prob  # adding two log-probabilities
        return sentence, total_probability

    def get_all_edits(self, word):
        """ Returns all of the possible edits that are in maximum 2-edit distance from the original word.

            Args:
                word (str): the word to get edits of.

            Return:
                set. All of the possible edits that are in maximum 2-edit distance from the original word.
                The format of the edit shall be a tuple of: (edited_word, prob),
                where prob is the log-probability of this error to occur.
                Note:
                    this set will contain only real words from the language-model's dictionary.
        """
        edits1 = self.edits1(word)
        # If it is a 2-letter word, than a 2-edit distance edit could just make it be any 2-letter word.
        # I will only check 2-edit distance of words with more than 2 letters:
        if len(word) > 2:
            all_edits = self.edits2(edits1)
        else:
            all_edits = edits1
        return self.keep_only_real_words(all_edits)

    def keep_only_real_words(self, all_edits):
        """ Returns all of the tuples which are of real vocabulary words.

            Args:
                all_edits (set): a set of all of the tuples to filter from.

            Return:
                set. All the filtered tuples which are of real vocabulary words.
                The format of the tuple shall be: (edited_word, prob),
                where prob is the log-probability of this error to occur.
                Note:
                    this set will contain only real words from the language-model's dictionary.
        """
        only_real_words = set()
        for edited_word, prob in all_edits:
            if edited_word and edited_word in self.vocabulary:
                only_real_words.add((edited_word, prob))
        return only_real_words

    def edits1(self, word):
        """ Returns all of the possible edits that are in 1-edit distance from the original word.

            Args:
                word (str): the word to get edits of.

            Return:
                set. All of the possible edits that are in 1-edit distance from the original word.
                The format of the edit shall be a tuple of: (edited_word, prob),
                where prob is the log-probability of this error to occur.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        insertions = [self.get_insertion(L, R) for L, R in splits if R]
        deletions = [self.get_deletion(L, R, c) for L, R in splits for c in letters]
        substitutions = [self.get_substitution(L, R, c) for L, R in splits if R for c in letters]
        transpositions = [self.get_transposition(L, R) for L, R in splits if len(R) > 1]

        all_edits1 = set(insertions + deletions + substitutions + transpositions)
        only_best_prob_edits = self.get_only_best_prob_edits(all_edits1)
        return only_best_prob_edits

    def edits2(self, edits1):
        """ Returns all of the possible edits that are in a distance of maximum 2-edit distance from the original word.

            Args:
                edits1 (set): the set of 1-edit-distance words, to make another edit distance of.
                    Each entry in the set is a tuple of: (edited_word, prob), as returned in the function self.edits1.

            Return:
                set. All of the possible edits that are in maximum 2-edit distance from the original word.
                The format of the edit shall be a tuple of: (edited_word, prob),
                where prob is the log-probability of this error to occur.
                Note:
                    this set shall contain both types of edits, 1-edit distance and 2-edit distance.
        """
        # Add both log-probabilities, because log(prob1 * prob2) = log(prob1) + log(prob2):
        all_edits2 = set((e2, prob1 + prob2) for e1, prob1 in edits1 for e2, prob2 in self.edits1(e1))
        # Take all edits into consideration:
        all_edits = edits1.union(all_edits2)
        only_best_prob_edits = self.get_only_best_prob_edits(all_edits)
        return only_best_prob_edits

    def get_insertion(self, L, R):
        """ Returns an insertion edit at the specified split

            Args:
                L (str): the left side of the word.
                R (str): the right side of the word.
                    R must have at least one letter in it (handled in the calling function).

            Return:
                A tuple in the following format: (edited_word, prob),
                where edited_word is the word created by this edit,
                and prob is the log-probability of this edit to occur.
        """
        if L:
            # if the left side of the word has at least one letter in it, take the last one in it:
            edit_entry = L[-1] + R[0]
        else:
            # else, it is an insertion at the start of the word, so add a # sign to represent it:
            edit_entry = '#' + R[0]
        return L + R[1:], self.get_prob_insertion(edit_entry)

    def get_deletion(self, L, R, c):
        """ Returns a deletion edit at the specified split and the specified deleted letter

            Args:
                L (str): the left side of the word.
                R (str): the right side of the word.
                c (char): the deleted letter from the word, which we want to add now.

            Return:
                A tuple in the following format: (edited_word, prob),
                where edited_word is the word created by this edit,
                and prob is the log-probability of this edit to occur.
        """
        if L:
            # if the left side of the word has at least one letter in it, take the last one in it:
            edit_entry = L[-1] + c
        else:
            # else, it is a deletion at the start of the word, so add a # sign to represent it:
            edit_entry = '#' + c
        return L + c + R, self.get_prob_deletion(edit_entry)

    def get_substitution(self, L, R, c):
        """ Returns a substitution edit at the specified split and the specified substituted letter

            Args:
                L (str): the left side of the word.
                R (str): the right side of the word.
                    R must have at least one letter in it (handled in the calling function).
                c (char): the substituted letter from the word, which we want to add now.

            Return:
                A tuple in the following format: (edited_word, prob),
                where edited_word is the word created by this edit,
                and prob is the log-probability of this edit to occur.
        """
        edit_entry = c + R[0]
        return L + c + R[1:], self.get_prob_substitution(edit_entry)

    def get_transposition(self, L, R):
        """ Returns a transposition edit at the specified split

            Args:
                L (str): the left side of the word.
                R (str): the right side of the word.
                    R must have at least two letters in it (handled in the calling function).

            Return:
                A tuple in the following format: (edited_word, prob),
                where edited_word is the word created by this edit,
                and prob is the log-probability of this edit to occur.
        """
        edit_entry = R[1] + R[0]
        return L + R[1] + R[0] + R[2:], self.get_prob_transposition(edit_entry)

    def get_prob_insertion(self, edit_entry):
        """ Returns the probability for an insertion error to occur.

            Args:
                edit_entry (str): the exact error that has occurred (two letters that represents the edit)

            Return:
                float. The log-probability of the error to occur.
        """
        return self.get_prob_edit('insertion', edit_entry, 'one_letter', edit_entry[0])

    def get_prob_deletion(self, edit_entry):
        """ Returns the probability for a deletion error to occur.

            Args:
                edit_entry (str): the exact error that has occurred (two letters that represents the edit)

            Return:
                float. The log-probability of the error to occur.
        """
        return self.get_prob_edit('deletion', edit_entry, 'two_letters', edit_entry)

    def get_prob_substitution(self, edit_entry):
        """ Returns the probability for a substitution error to occur.

            Args:
                edit_entry (str): the exact error that has occurred (two letters that represents the edit)

            Return:
                float. The log-probability of the error to occur.
        """
        return self.get_prob_edit('substitution', edit_entry, 'one_letter', edit_entry[1])

    def get_prob_transposition(self, edit_entry):
        """ Returns the probability for a transposition error to occur.

            Args:
                edit_entry (str): the exact error that has occurred (two letters that represents the edit)

            Return:
                float. The log-probability of the error to occur.
        """
        return self.get_prob_edit('transposition', edit_entry, 'two_letters', edit_entry)

    def get_prob_edit(self, error_type, edit_entry, denominator_type, denominator_entry):
        """ Returns the probability for an error to occur.

            Args:
                error_type (str): the type of error that happened.
                    Can be one of the following: ['insertion' / 'deletion' / 'substitution' / 'transposition']
                edit_entry (str): the exact error that has occurred (two letters that represents the edit)
                denominator_type (str): the type of denominator that this error will use.
                    Can be one of the following: ['one_letter' / 'two_letters']
                denominator_entry (str): the entry to the denominator. It differs in every type of error.

            Return:
                float. The log-probability of the error to occur.
        """
        if denominator_entry not in self.denominator_error_tables[denominator_type]:
            return math.log(1)
        # Smoothing:
        if edit_entry not in self.error_tables[error_type] or self.error_tables[error_type][edit_entry] == 0:
            # return math.log(1 / (len(self.denominator_error_tables[denominator_type]) + 1))
            return math.log(1 / self.denominator_error_tables[denominator_type][denominator_entry])
        probability = self.error_tables[error_type][edit_entry] / self.denominator_error_tables[denominator_type][denominator_entry]
        log_prob = math.log(probability)
        return log_prob

    def get_only_best_prob_edits(self, all_edits):
        """ Keeps only the edits with the highest probability from the given set of edits

            Args:
                all_edits (set): the set of edits we want to subset from.

            Return:
                set. The subset of edits, containing only the best probability for each edited word.
        """
        # Create a dictionary containing the max log-probability found for each word in the given set:
        only_best_prob_edits_dict = {}
        for edit, prob in all_edits:
            if edit not in only_best_prob_edits_dict.keys():
                only_best_prob_edits_dict[edit] = prob
            else:
                if only_best_prob_edits_dict[edit] < prob:
                    only_best_prob_edits_dict[edit] = prob
        # Turn this dictionary to a set of tuples:
        only_best_prob_edits_set = set((edit, prob) for edit, prob in only_best_prob_edits_dict.items())
        return only_best_prob_edits_set


def who_am_i():  # this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Rotem Lev Lehman', 'id': '208965814', 'email': 'levlerot@post.bgu.ac.il'}

# The following class is the language model class I have submitted to ex1.
# I am submitting it as well, because I need to use it in the build_model function:


class Ngram_Language_Model:
    """The class implements a Markov Language Model that learns a model from a given text.
        It supports language generation and the evaluation of a given string.
        The class can be applied on both word level and character level.
    """

    def __init__(self, n=3, chars=False):
        """Initializing a language model object.
        Args:
            n (int): the length of the markov unit (the n of the n-gram). Defaults to 3.
            chars (bool): True iff the model consists of ngrams of characters rather than word tokens.
                          Defaults to False
        """
        self.n = n
        self.chars = chars
        self.ngram_counter = None  # this counter will help us for the generation of the language model
        self.ngram_dictionary = None  # this dictionary will map from a context to the possible grams
        self.context_counter = None  # this counter will count for each context how many times it appears
        self.grams_set = None  # this set will hold all of the unique grams in the given text
        self.next_gram_choosing_dict = None  # this dictionary will hold for each context, both: (1) a list of possible next grams, and (2) a list of their corresponding probabilites
        self.all_contexts = None  # this list will hold all of the contexts in the text (for choosing the initial context)
        self.all_contexts_probs = None  # this list will hold the probabilities of each context
        self.ngrams_set = None  # this set will hold all unique n-grams in the given text
        self.sum_contexts = None  # this represents the amount of total contexts in the text
        self.num2counter_and_sum = None  # this dictionary will map from an amount of grams, to (1) a counter for this amount of grams, and to (2) the amount of the total appearances

    def build_model(self, text):  # should be called build_model
        """populates a dictionary counting all ngrams in the specified text.

            Args:
                text (str): the text to construct the model from.
        """
        split_text = self.get_tokens(text)
        all_ngrams = self.get_all_ngrams(split_text, self.n)

        self.num2counter_and_sum = {}
        self.ngram_counter = Counter(all_ngrams)  # count each n-gram and see how many times it occurred in the list.
        self.num2counter_and_sum[self.n] = self.ngram_counter

        for i in range(1, self.n):
            # for each amount of grams, get all of the i-grams and put them into the num2counter_and_sum dictionary:
            all_i_grams = self.get_all_ngrams(split_text, i)
            curr_counter = Counter(all_i_grams)
            curr_sum = len(all_i_grams)
            self.num2counter_and_sum[i] = (curr_counter, curr_sum)

        self.grams_set = set(split_text)

        all_n_minus1_grams = self.get_all_ngrams(split_text, self.n - 1)
        self.context_counter = Counter(all_n_minus1_grams)

        self.ngram_dictionary = {}
        self.next_gram_choosing_dict = {}
        self.ngrams_set = set()
        for grams, num in self.ngram_counter.items():  # for each n-gram, map from the n-1 first grams to the last gram.
            self.ngrams_set.add(grams)
            context, last_gram = self.split_ngram(grams)

            # add to the ngram_dictionary:
            if context not in self.ngram_dictionary.keys():
                self.ngram_dictionary[context] = {}
            self.ngram_dictionary[context][last_gram] = num

        for context, possible_grams in self.ngram_dictionary.items():
            total_possibilities = sum(possible_grams.values())
            only_grams = [x for x in possible_grams.keys()]
            prob_dist = [possible_grams[k] / total_possibilities for k in possible_grams.keys()]
            self.next_gram_choosing_dict[context] = (only_grams, prob_dist)

        self.sum_contexts = sum(self.context_counter.values())
        self.all_contexts = [k for (k, v) in self.context_counter.items()]
        self.all_contexts_probs = [v / self.sum_contexts for (k, v) in self.context_counter.items()]

    def get_tokens(self, text):
        """Returns a list of tokens from the given text.
        Tokens are split either by space or to a list of characters, depending on the value of self.chars.

            Args:
                text (str): the text we wish to tokenize.

            Return:
                List. The list of tokens.

        """
        if self.chars:
            split_text = list(text)  # if we choose to use n-grams of characters, than we shall split each character to different list entry.
        else:
            split_text = text.split(' ')  # split the text by space, so each word will be in a different list entry.
        return split_text

    def get_all_ngrams(self, split_text, n):
        """Returns a list of n-grams from the given split text.

            Args:
                split_text (list): the list of all tokens.

            Return:
                List. The list of all n-grams.

        """
        return [" ".join(split_text[i:i + n]) for i in range(len(split_text) - n + 1)]

    def get_model(self):
        """Returns the model as a dictionary of the form {ngram:count}
        """
        return dict(self.ngram_counter.items())

    def generate(self, context=None, n=20):
        """Returns a string of the specified length, generated by applying the language model
        to the specified seed context. If no context is specified the context should be sampled
        from the models' contexts distribution. Generation should stop before the n'th word if the
        contexts are exhausted.

            Args:
                context (str): a seed context to start the generated string from. Defaults to None
                n (int): the length of the string to be generated.

            Return:
                String. The generated text.

        """
        if context is not None:
            normalized_context = normalize_text(context)
            split_context = self.get_tokens(normalized_context)
            curr_context = " ".join(split_context)
        else:  # context is None:
            curr_context = self.sample_initial_context()

        curr_context_list = curr_context.split(' ')
        sentence_grams = []
        for gram in curr_context_list:
            sentence_grams.append(gram)

        if len(curr_context_list) > self.n - 1:
            curr_context_list = curr_context_list[(len(curr_context_list) - self.n + 1):]  # if the context is longer than n-1, only proceed with the last n-1 words in the context.
            curr_context = " ".join(curr_context_list)

        if curr_context not in self.ngram_dictionary.keys():
            # The given context does not exists:
            curr_context = self.sample_initial_context()
            curr_context_list = curr_context.split(' ')
            sentence_grams = []
            for gram in curr_context_list:
                sentence_grams.append(gram)

        while len(sentence_grams) < n:
            # get next gram in the sentence:
            next_gram = self.sample_next_gram(curr_context)
            if next_gram is None:  # if we have never seen this context before, return the sentence as it is until now.
                break
            sentence_grams.append(next_gram)

            # fix current context:
            if self.n > 1:  # fix only if there is a point to doing so, if it is a model without contexts, than we do not need to fix it...
                del curr_context_list[0]
                curr_context_list.append(next_gram)
                curr_context = " ".join(curr_context_list)

        return " ".join(sentence_grams)  # create a string from the list of the grams in the sentence

    def sample_initial_context(self):
        """Samples an initial context to start a new sentence
        """
        return choices(self.all_contexts, self.all_contexts_probs)[0]

    def sample_next_gram(self, context):
        """Returns the next gram using the given context. We choose the next gram using sampling from a distribution.

            Args:
                context (str): the context that we want to find the next gram from.

            Returns:
                String. The next gram we chose.
        """
        if context not in self.next_gram_choosing_dict.keys():
            return None  # If the context does not exist in the model, return None
        possible_ngrams, prob_dist = self.next_gram_choosing_dict[context]
        return choices(possible_ngrams, prob_dist)[0]

    def evaluate(self, text):
        """Returns the log-likelihod of the specified text to be generated by the model.
           Laplace smoothing should be applied if necessary.

           Args:
               text (str): Text to ebaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """
        nt = normalize_text(text)
        split_text = self.get_tokens(nt)
        all_ngrams = self.get_all_ngrams(split_text, self.n)

        # go over all ngrams, and for each ngram, calculate the probability of it appearing.
        # multiply all of those probabilities. We can do that because of the Markov chain rule:
        probability = 0
        for ngram in all_ngrams:  # log(p1 * p2 * ... * pn) is equal to: log(p1) + log(p2) + ... + log(pn)
            probability += math.log(self.get_prob(ngram))  # send the n-gram to a function that gets it's probability (or smoothed probability if it does not exist...)
        probability += self.get_log_prob_context(split_text[:self.n - 1])
        return probability

    def get_log_prob_context(self, context_list):
        """Returns the probability of the specified context.
        If the context exists in the context set, than use the regular probability, else, smooth it's probability.

            Args:
                context_list (list): the context to have it's probability

            Returns:
                float. The probability.
        """
        log_prob = 0
        prev_context = ''
        for i in range(1, len(context_list) + 1):
            curr_context = " ".join(context_list[:i])
            curr_counter, curr_sum = self.num2counter_and_sum[i]
            prev_amount_of_possibilities = curr_sum
            if i > 1:
                prev_counter, _ = self.num2counter_and_sum[i-1]
                if prev_context in prev_counter.keys():
                    prev_amount_of_possibilities = prev_counter[prev_context]

            if curr_context in curr_counter.keys():
                # The context exists in the text:
                log_prob += math.log(curr_counter[curr_context] / prev_amount_of_possibilities)
            else:
                # The context does not exist in the text, so we need to smooth it:
                V = len(self.grams_set)  # number of unique words (grams) in the text - vocabulary's size.
                log_prob += math.log(1.0 / V)
            prev_context = curr_context
        return log_prob

    def get_prob(self, ngram):
        """Returns the probability of the specified ngram.
        If the ngram exists in the n-grams set, than use the regular probability, else, smooth it's probability.

            Args:
                ngram (str): the ngram to have it's probability

            Returns:
                float. The probability.
        """
        context, last_gram = self.split_ngram(ngram)
        if context in self.ngram_dictionary.keys() and last_gram in self.ngram_dictionary[context].keys():
            # The ngram exists in the text:
            return self.calc_regular_probability(ngram)
        else:
            # The ngram does not exist in the text, so we need to smooth it:
            return self.smooth(ngram)

    def calc_regular_probability(self, ngram):
        """Returns the regular (non-smoothed) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability

            Returns:
                float. The probability.
        """
        context, last_gram = self.split_ngram(ngram)
        C_context = self.context_counter[context]
        C_ngram = self.ngram_dictionary[context][last_gram]
        Prob = C_ngram / C_context
        return Prob

    def smooth(self, ngram):
        """Returns the smoothed (Laplace) probability of the specified ngram.

            Args:
                ngram (str): the ngram to have it's probability smoothed

            Returns:
                float. The smoothed probability.
        """
        context, last_gram = self.split_ngram(ngram)
        V = len(self.grams_set)  # number of unique words (grams) in the text - vocabulary's size.
        C_context = 0
        C_ngram = 0
        if context in self.context_counter.keys():  # if the context exists:
            C_context = self.context_counter[context]
        if context in self.ngram_dictionary.keys() and last_gram in self.ngram_dictionary[context].keys():  # if the ngram exists:
            C_ngram = self.ngram_dictionary[context][last_gram]
        P_laplace = (C_ngram + 1) / (C_context + V)  # the formula for the Laplace smoothing
        return P_laplace

    def split_ngram(self, ngram):
        """Splits the specified ngram to it's context and last-gram.

            Args:
                ngram (str): the ngram to split

            Returns:
                Tuple. The context and the last gram
        """
        ngram_list = ngram.split(' ')
        context = " ".join(ngram_list[:-1])
        last_gram = ngram_list[-1]
        return context, last_gram


def normalize_text(text, lower_text=True, pad_punctuations=True):
    """Returns a normalized string based on the specifiy string.
       You can add default parameters as you like (they should have default values!)
       You should explain your decitions in the header of the function.

       Args:
           text (str): the text to normalize
           lower_text (bool): specifies if we want to lower-case the given text. Defaults to True
           pad_punctuations (bool): specifies if we want to pad punctuations with white-spaces - which means that they will be grams. Defaults to True

       Returns:
           string. the normalized text.
    """
    edited_text = text
    if lower_text:
        edited_text = text.lower()
    if pad_punctuations:
        edited_text = " ".join(re.findall(r'\w+|[.,/#!$%^&*;:{}=\-_`~()\[\]]', edited_text, flags=re.VERBOSE))
    return edited_text
