import numpy as np
import pandas as pd
from itertools import combinations

obs = 1000
x = np.random.random_integers(0, 500, (obs, 1))
y = np.random.random_integers(1, 5, (obs, 1))
tg = np.zeros((obs, 1))
# t[(y == 2) | (y == 5)] = 1
tg[x > 250] = 1
X = pd.DataFrame(np.hstack([x, y]))
X.columns = ["feature_X", "feature_Y"]
Y = pd.Series(tg.reshape(len(tg)))
Y.columns = ["dependiente"]


class DecisionNode:
    def __init__(self, features, target, min_split_samples = 80):
        """ Creates a decision node.

        The DecisionNode object performs all the actions necessary for a node to split.

        Attributes
        -------------
            self.X: Is a Pandas DataFrame of NxK. N observations for K features.

            self.Y: Is a Pandas DataFrame of Nx1. N observation for a Target Variable.

            self.categorical: If true, it means that the best split is done with a categorical variable.

            self.rule: Determines the splitting rule (only available for continuous features).

            self.left_rule: Determines the splitting rule (the subset of elements). Only available for categorical
                            features.

            self.right_rule: Determines the splliting rule (the other subset of elements). Only available for
                             categorical features.

            self.feature_name: The name of the feature that provides the best split overall.

            self.gini_index: The Gini Index of the best split overall.

            self.counts: The counts of the targets in the node.

            self.predicted_class: The predicted class of the node. [[INCLUDE]]

            self.left_branch_X: All the observations (features) that follow the best split rule.

            self.right_branch_X: All the observatios (features) that don't follow the best split rule.

            self.left_branch_Y: All the observations (target) that follow the best split rule.

            self.right_branch_Y: All the observations (target) that don't follow the best split rule.

            self.left_counts: The counts of the target variable in the left child node.

            self.right_counts: The counts of the target variable in the right child node.

            self.left_predicted: What class does the left child node predict.

            self.right_predicted: What class does the right child node predict.
            
            self.left_gini: Returns the Gini Index of the left part.
            
            self.right_gini: Returns the Gini Index of the right part.
            
            self.left_sample: Returns N that goes to the left child.
            
            self.right_sample: Returns N that goes to the right child.


        """

        self.X = features
        self.Y = target
        self.categorical = False
        self.rule = None
        self.left_rule = None
        self.right_rule = None
        self.feature_name = ""
        self.gini_index = None
        self.counts = self.get_counts(self.Y)

        self.predicted_class = self.predict_class(self.counts)
        self.left_branch_X = None
        self.right_branch_X = None
        self.left_branch_Y = None
        self.right_branch_Y = None

        self.left_counts = {}
        self.right_counts = {}
        self.left_predicted = None
        self.right_predicted = None
        self.left_gini = None
        self.right_gini = None
        self.left_sample = None
        self.right_sample = None

        self.min_split_samples = min_split_samples
        self.structure = {}
        self.left = ""
        self.right = ""

    @staticmethod
    def best_split(given_set):
        return min(given_set, key=given_set.get)

    @staticmethod
    def gini(given_set):
        n = 0
        for i in given_set:
            n += given_set[i]

        p = 1
        for i in given_set:
            p -= (given_set[i] / n) ** 2

        return p, n

    def split_impurity(self, outcomes, iterables, target, feature, is_combination=False):
        impur_l = {}  # Sacará la impureza de cada nodo.
        impur_r = {}  # Calcula impureza

        if not is_combination:
            for t in iterables:
                lsplt = {}  # Van a contar cuántas observaciones hay de cada valor en menor
                rsplt = {}  # Y en mayor

                less = target[feature <= t]
                great = target[feature > t]

                for o in outcomes:
                    lsplt[o] = list(less).count(o)
                    rsplt[o] = list(great).count(o)

                impur_l[t] = self.gini(lsplt)
                impur_r[t] = self.gini(rsplt)

        else:
            for t in iterables:
                criteria = list(t)

                lsplt = {}
                rsplt = {}
                follow_criteria = np.in1d(feature, criteria)
                follow = target[follow_criteria]
                not_follow = target[~follow_criteria]

                for o in outcomes:
                    lsplt[o] = list(follow).count(o)
                    rsplt[o] = list(not_follow).count(o)

                impur_l[t] = self.gini(lsplt)
                impur_r[t] = self.gini(rsplt)

        split_impurity = {}
        for i in iterables:
            n = impur_l[i][1] + impur_r[i][1]
            pr = impur_r[i][1] / n
            pl = impur_l[i][1] / n
            split_impurity[i] = pr * impur_r[i][0] + pl * impur_l[i][0]

        return split_impurity

    @staticmethod
    def select_combinations(vals):
        comb = []
        e = len(vals)
        if e > 30:
            pass
        else:
            if e % 2 == 0:
                end = int(e / 2)
                for i in range(1, end + 1):
                    if i == end:
                        cmb = list(combinations(vals, i))
                        enough = int(len(cmb) / 2)
                        comb.extend(cmb[:enough])
                    else:
                        comb.extend(combinations(vals, i))

            else:
                end = int((e - 1) / 2)
                for i in range(1, end + 1):
                    comb.extend(combinations(vals, i))

            return comb

    @staticmethod
    def get_counts(target):
        targets = set(target)
        counts = {}
        for yi in targets:
            counts[yi] = list(target).count(yi)

        return counts

    @staticmethod
    def predict_class(counts):
        p_class = max(counts, key=counts.get)  # Selects the key. Maybe it's even, so it returns it.
        count = counts[p_class]  # Picks the value
        if len(counts) == len(set(counts.values())):
            return p_class
        else:
            res = []

            for key, value in counts.items():
                if value == count:
                    res.append(key)

            return res

    def split(self, feature):
        """ This function figures out where's located the best split of the feature.

        Parameters
        ------------
            feature: Is the k-th feature available.

        Returns
        ------------
            Depending on the type of feature (ordinal or categorical), the output will be:
            Categorical (Less than 5 categories): [Gini Impurity of the Best Split, Combination 1, Combination 2]
            Ordinal (More than 5 categories):     [Gini Impurity of the Best Split, Threshold]

        """
        outcomes = list(set(self.Y.unique()))

        values = list(set(feature))  # Cats

        if len(values) == 1:  # There's no way to split the data...
            return 105, 0, 0, False
        elif len(values) <= 5:  # Categorical features

            opts = self.select_combinations(values)  # Combinations
            split_imp = self.split_impurity(outcomes, opts, self.Y, feature, True)
            part1 = self.best_split(split_imp)  # Best Split
            part2 = list(np.setdiff1d(values, list(part1)))

            return split_imp[part1], list(part1), part2, True  # Returns impurity & combinations, and True (categ)

        else:  # Continuous and ordinal
            tmp = set(feature)
            n_elem = len(tmp)
            if n_elem < 100:
                percentiles = list(tmp)
                percentiles.sort()
                percentiles = percentiles[:(len(tmp) - 1)]
                del tmp

            else:
                del tmp
                percentiles = [np.percentile(feature, i) for i in range(100)]

            split_imp = self.split_impurity(outcomes, percentiles, self.Y, feature)  # Node's impurity
            threshold_optimo = self.best_split(split_imp)
            return split_imp[threshold_optimo], threshold_optimo, 0, False  # impurity, Returns T* and False

    def select_feature(self):
        """ Performs the best split for every feature available.

        Returns
        ----------
            Returns the name of the feature that provides with the best split overall, the Gini Impurity of the Split
            and the decision rule of the split.

        """
        feat = {}
        imps = {}
        for f in self.X:
            feat[f] = self.split(self.X.ix[:, f])
            imps[f] = feat[f][0]

        self.feature_name = min(imps, key=imps.get)
        self.categorical = feat[self.feature_name][3]  # If the variable is categorical, the 4th element is True.
        self.gini_index = feat[self.feature_name][0]
        if self.categorical:
            self.left_rule, self.right_rule = feat[self.feature_name][1:3]
        else:
            self.rule = feat[self.feature_name][1]

    def data_split(self):
        if self.categorical:

            condition = self.X.ix[:, self.feature_name].isin(self.left_rule)
        else:
            condition = self.X[self.feature_name] <= self.rule

        self.left_branch_X, self.left_branch_Y = self.X[condition], self.Y[condition]
        self.right_branch_X, self.right_branch_Y = self.X[~condition], self.Y[~condition]

    def compute_split_counts(self):
        self.left_counts = self.get_counts(self.left_branch_Y)
        self.right_counts = self.get_counts(self.right_branch_Y)
        self.left_predicted = self.predict_class(self.left_counts)
        self.right_predicted = self.predict_class(self.right_counts)
        self.left_gini, self.left_sample = self.gini(self.left_counts)
        self.right_gini, self.right_sample = self.gini(self.right_counts)

    def start(self):
        self.select_feature()
        self.data_split()
        self.compute_split_counts()

# data = pd.read_csv(r"C:\Users\JuanEduardo\Google Drive\Machine Learning\train.csv")
# data = pd.DataFrame(data)
# data = data.dropna()
# Y = data.ix[:, "Survived"]
# variables = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
# X = data[variables]
# X.ix[X.ix[:, "Sex"] == "male", "Sex"] = 1
# X.ix[X.ix[:, "Sex"] == "female", "Sex"] = 0

my_node = DecisionNode(X, Y)
my_node.start()
