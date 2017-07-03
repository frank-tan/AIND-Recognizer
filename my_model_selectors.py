import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model_bic_dict = {}
        # for every possible number of hidden states parameter
        for hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model: GaussianHMM = self.base_model(hidden_states)
                log_l = model.score(self.X, self.lengths)
            except:
                log_l = float("-inf")

            # data points is the number of rows
            data_points = self.X.shape[0]

            # calculation for parameters
            states = hidden_states
            features = self.X.shape[1]
            starting_probs = states - 1
            transition_probs = states * (states - 1)
            means = states * features
            variance = states * features
            params = starting_probs + transition_probs + means + variance
            bic = -2 * log_l + params * np.log(data_points)
            model_bic_dict[model] = bic

        assert len(model_bic_dict.keys()) > 0
        # return the model which has the lowest bic
        return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model_dic_dict = {}

        # for every possible number of hidden states parameter
        for hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model: GaussianHMM = self.base_model(hidden_states)
                log_l = model.score(self.X, self.lengths)

            except:
                log_l = float("-inf")

            sum_all_other_log_l = 0
            m = 0

            # for each other word
            for other_word in self.hwords.keys():
                if self.this_word != other_word:
                    other_word_x, other_word_lengths = self.hwords[other_word]
                    # score the model with other words and add it to the sum
                    try:
                        other_word_score = model.score(other_word_x, other_word_lengths)
                    except:
                        other_word_score = float("+inf")
                    sum_all_other_log_l += other_word_score
                    m += 1
            assert m > 1
            dic = log_l - sum_all_other_log_l / (m - 1)

            # calculate dic for the number of hidden states
            model_dic_dict[model] = dic

        # return the model which has the highest dic
        return max(model_dic_dict, key=model_dic_dict.get)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        n_data_sets = len(self.sequences)
        if n_data_sets < 2:
            return self.base_model(self.n_constant)

        n_splits = min(n_data_sets, 3)
        split_method = KFold(n_splits)
        hidden_states_score_dict = {}

        # for every possible number of hidden states parameter
        for hidden_states in range(self.min_n_components, self.max_n_components + 1):
            split_training_score_array = []

            # split the data into training sets and testing sets
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                x_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)

                try:
                    # create the model with training sets
                    model: GaussianHMM = GaussianHMM(n_components=hidden_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(x_train, lengths_train)

                    # calculate the score of the model with testing sets
                    split_training_score_array.append(model.score(x_test, lengths_test))
                except:
                    continue

            # record the average score for the specific hidden states parameter
            if len(split_training_score_array) > 0:
                hidden_states_score_dict[hidden_states] = np.average(split_training_score_array)

        # find the optimal hidden states which gives highest average score
        optimal_hidden_states = max(hidden_states_score_dict, key=hidden_states_score_dict.get)

        # use the optimal hidden states parameter to train a new model with all data
        return self.base_model(optimal_hidden_states)
