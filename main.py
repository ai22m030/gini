import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings(action='ignore')


class Node:
    def __init__(self, feat, tar):
        self.features = feat
        self.target = tar
        self.feature_types = self.select_types()

    # Check feature types
    def select_types(self):
        feature_types = []
        for item in self.features.columns:
            if self.features[item].dtype != "O":
                feature_types.append((item, "numerical"))
            else:
                if self.features[item].nunique() <= 2:
                    feature_types.append((item, "binary"))
                else:
                    feature_types.append((item, "multiclass"))
        return feature_types

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
        gini_node = []
        combinations = []

        for i in range(1, self.features[feature].nunique()):
            combinations = combinations + list(itertools.combinations(self.features[feature].unique(), i))

        for item in combinations:
            t1 = self.target[self.features[feature].isin(item)]
            t2 = self.target[~self.features[feature].isin(item)]
            args = t1.value_counts().tolist() + t2.value_counts().tolist()
            gini_node.append(Node.gini_impurity_total(*args))

        return gini_node, combinations  # Return all the values

    # Get the best gini values for each feature
    def check_nodes(self):
        gini_values = []
        for feature in self.features.columns:
            calculated_gini, combinations = self.calculate_gini(feature)
            best_combination = combinations[np.argmin(calculated_gini)]
            gini_values.append((feature, best_combination, np.min(calculated_gini)))
        return gini_values

    # Get best node
    def best_node(self):
        gini_values = self.check_nodes()
        values = [item[2] for item in gini_values]
        node_gini = Node.gini_impurity(*self.target.value_counts().tolist())

        if node_gini > np.min(values):
            best_feature = gini_values[np.argmin(values)][0]
            best_combination = gini_values[np.argmin(values)][1]
            print(f"Best node {best_feature} and yes/no decision {best_combination[0]}")


titanic = pd.read_csv("train.csv")
target = titanic["Survived"]
features = titanic.loc[:, ["Pclass", "Sex", "Embarked"]]
features.Embarked.fillna(features.Embarked.mode()[0], inplace=True)
features.Pclass = features.Pclass.map({1: "1st", 2: "2nd", 3: "3rd"})

# Reading and cleaning the data
root = Node(features, target)

# Building the tree
root.best_node()
