import math
import re
from collections import Counter
from random import choices


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


def who_am_i():
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Rotem Lev Lehman', 'id': '208965814', 'email': 'levlerot@post.bgu.ac.il'}
