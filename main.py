import numpy as np
import pandas as pd
import itertools


class Gini:
    def __init__(self, feat, tar):
        self.features = feat
        self.target = tar

    # Calculate gini impurity for each feature at a node
    @staticmethod
    def gini_impurity_total(a=0, b=0, c=0, d=0):
        total_elements = a + b + c + d
        gini_1 = 1 - np.square(a / (a + b)) - np.square(b / (a + b))
        gini_2 = 1 - np.square(c / (c + d)) - np.square(d / (c + d))
        total_gini = ((a + b) / total_elements) * gini_1 + ((c + d) / total_elements) * gini_2
        return total_gini

    @staticmethod
    def gini_impurity(a=0, b=0):
        return 1 - np.square(a / (a + b)) - np.square(b / (a + b))

    # Calculate gini for all feature combinations in categorical features
    def calculate_gini(self, feature):
        gini_values = []
        combinations = []

        for i in range(1, self.features[feature].nunique()):
            combinations = combinations + list(itertools.combinations(self.features[feature].unique(), i))

        for item in combinations:
            t1 = self.target[self.features[feature].isin(item)]
            t2 = self.target[~self.features[feature].isin(item)]
            args = t1.value_counts().tolist() + t2.value_counts().tolist()
            gini_values.append(Gini.gini_impurity_total(*args))

        return gini_values, combinations  # Return all the values

    # Get the best gini values for each feature
    def check_features(self):
        gini_values = []
        for feature in self.features.columns:
            calculated_gini, combinations = self.calculate_gini(feature)
            best_combination = combinations[np.argmin(calculated_gini)]
            gini_values.append((feature, best_combination, np.min(calculated_gini)))
        return gini_values

    # Get best node
    def best_feature(self):
        gini_values = self.check_features()
        values = [item[2] for item in gini_values]

        best_feature = gini_values[np.argmin(values)][0]
        best_combination = gini_values[np.argmin(values)][1]
        print(f"Best node {best_feature} and yes/no decision {best_combination[0]}")


titanic = pd.read_csv("train.csv")
target = titanic["Survived"]
features = titanic.loc[:, ["Pclass", "Sex", "Embarked"]]
features.Embarked.fillna(features.Embarked.mode()[0], inplace=True)
features.Pclass = features.Pclass.map({1: "1st", 2: "2nd", 3: "3rd"})

# Reading data
root = Gini(features, target)

root.best_feature()
