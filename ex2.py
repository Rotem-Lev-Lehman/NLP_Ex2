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

    def add_language_model(self, lm):
        """Adds the specified language model as an instance variable.
            (Replaces an older LM disctionary if set)

            Args:
                lm: a language model object
        """

    def learn_error_tables(self, errors_file):
        """Returns a nested dictionary {str:dict} where str is in:
            <'deletion', 'insertion', 'transposition', 'substitution'> and the
            inner dict {str: int} represents the confution matrix of the
            specific errors, where str is a string of two characters mattching the
            row and culumn "indixes" in the relevant confusion matrix and the int is the
            observed count of such an error (computed from the specified errors file).
            Examples of such string are 'xy', for deletion of a 'y'
            after an 'x', insertion of a 'y' after an 'x'  and substitution
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

    def add_error_tables(self, error_tables):
        """ Adds the speficied dictionary of error tables as an instance variable.
            (Replaces an older value disctionary if set)

            Args:
                error_tables (dict): a dictionary of error tables in the format
                returned by  learn_error_tables()
        """


    def evaluate(self,text):
        """Returns the log-likelihod of the specified text given the language
            model in use. Smoothing is applied on texts containing OOV words

           Args:
               text (str): Text to evaluate.

           Returns:
               Float. The float should reflect the (log) probability.
        """

    def spell_check(self, text, alpha):
        """ Returns the most probable fix for the specified text. Use a simple
            noisy channel model is the number of tokens in the specified text is
            smaller than the length (n) of the language model.

            Args:
                text (str): the text to spell check.
                alpha (float): the probability of keeping a lexical word as is.

            Return:
                A modified string (or a copy of the original if no corrections are made.)
        """


def who_am_i(): #this is not a class method
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'John Doe', 'id': '012345678', 'email': 'jdoe@post.bgu.ac.il'}
