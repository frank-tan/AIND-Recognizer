import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # for each word in the testing set
    for word_index, _ in test_set.get_all_Xlengths().items():
        x, length = test_set.get_item_Xlengths(word_index)
        word_log_l_dict = {}
        # try the word on every model and score the probabilities of matching
        for word, model in models.items():
            try:
                word_log_l_dict[word] = model.score(x, length)
            except:
                word_log_l_dict[word] = float("-inf")

        probabilities.append(word_log_l_dict)
        guesses.append(max(word_log_l_dict, key=word_log_l_dict.get))

    return probabilities, guesses
