import numpy as np
import pandas as pd


class DecisionNode(object):
    '''
    A class that represents a node in a decision tree.
    It stores the question being considered in the form
    of [feature_index, feature_value] and stores a pointer
    to the nodes that result from answering true or false.
    '''

    def __init__(self, question, true, false):
        self.question = question
        self.true = true
        self.false = false

        self.true_labels = None
        self.false_labels = None


class DecisionTree(object):
    '''
    A class that represents a decision tree classifier.
    '''

    def __init__(self):
        self.decisions = []
        self.decision_number = 0

        self.root = None

    def select_feature(self, features, feature_index):
        '''
        Given a feature index, selects all the instances
        of that type of feature in the data set.

        Arguments:
            features: the features in the data set
            feature_index: the index of the feature to extract

        Returns:
            selected_feature: the selected feature from features.
        '''

        selected_feature = [row[feature_index] for row in features]
        return selected_feature

    def unique_features(self, features, feature_index):
        '''
        Finds the unique values of a specific feature.

        Arguments:
            Arguments:
            features: the features in the data set
            feature_index: the index of the feature to extract

        Returns:
            unique: the unique values in the selected feature.
        '''
        unique = set(self.select_feature(features, feature_index))
        return unique

    def gini_impurity(self, labels):
        '''
        Given some set of labels, calculates the gini impurity: a measure
        of how often a randomly chosen label would be incorrectly labelled i.e.
        the probability of obtaining two different outputs.

        Arguments:
            labels: the set of labels under consideration.

        Returns:
            purity: the gini impurity of the labels.
        '''
        number_of_data = len(labels)
        classes = set(labels)

        class_dict = {}

        for c in classes:
            class_dict[c] = 0

        for label in labels:
            class_dict[label] += 1 / number_of_data

        priors = list(class_dict.values())

        purity = 1 - sum([p**2 for p in priors])

        return purity

    def info_gain(self, true_labels, false_labels, current_uncertainty):
        '''
        Calculates the information gain: the change in weighted gini 
        purity from a current state to another state (after a question
        has been asked).

        Arguments:
            true_labels: the resulting labels if answering true to a decision.
            false_labels: the resulting labels if answering false to a decision.
            current_uncertainty: the gini impurity before a decision has been made.

        Returns:
            gain: the info gain
        '''
        p = float(len(true_labels)) / (len(true_labels) + len(false_labels))
        gain = current_uncertainty - p * self.gini_impurity(true_labels) - (1 - p) * self.gini_impurity(false_labels)
        return gain

    def fit(self, features, labels, decision_node=None, decision=None):
        '''
        Trains a decision tree, given some features and labels of a data set.
        This method is called recursively to construct a tree of most relevant
        questions to ask in order to classify the data set.

        Arguments:
            features: the features of the dataset.
            labels: the labels of the dataset.
            decision_node: the decision node of the current decision.
            decision: whether a true or false decision was made.
        '''
        number_of_data = len(features)
        number_of_features = len(features[0])

        # Calculate the gini impurity of the labels
        uncertainty = self.gini_impurity(labels)

        # Keep track of the best gain through asking different questions
        best_gain = -1

        # For each type of feature in the data set
        for j in range(number_of_features):
            unique_feat = self.unique_features(features, j)

            # For each value of each feature in the dataset
            for k, feat in enumerate(unique_feat):
                true_labels = []
                true_features = []
                false_labels = []
                false_features = []

                # For each data point in the dataset
                for i in range(number_of_data):

                    # Make a decision on this data point using
                    # the specific feature and specific value
                    if isinstance(feat, str):
                        condition = features[i][j] == feat

                    else:
                        condition = features[i][j] >= feat

                    # If true, append to true arrays
                    if condition:
                        true_labels.append(labels[i])
                        true_features.append(features[i])

                    # If false, append to false arrays
                    else:
                        false_labels.append(labels[i])
                        false_features.append(features[i])

                # Calculate the information gain by making these decisions.
                gain = self.info_gain(true_labels, false_labels, uncertainty)

                # Keep track of best info gain
                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = j
                    best_feature = feat
                    best_true_features = true_features
                    best_false_features = false_features
                    best_true_labels = true_labels
                    best_false_labels = false_labels

        # Make a decision node that stores the information of this
        # current node and the best decision to make.
        next_decision_node = DecisionNode(
            [best_feature_index, best_feature], None, None)
        next_decision_node.true_labels = best_true_labels
        next_decision_node.false_labels = best_false_labels

        # If the tree is first instantiated, set this node to be the root.
        if decision is None:
            self.root = next_decision_node

        # If the decision has a positive effect on the gain, 
        # link the decision into a tree structure.
        if gain > 0:
            if decision == True:
                decision_node.true = next_decision_node

            if decision == False:
                decision_node.false = next_decision_node

        # If the decision made was true, recursively call
        # this method again to find the next best decision
        # for the True decision.
        decision = True
        if len(best_true_labels) > 1:
            if gain > 0:
                self.fit(best_true_features, best_true_labels,
                         next_decision_node, decision)

        # If the decision made was false, recursively call
        # this method again to find the next best decision
        # for the False decision.
        decision = False
        if len(best_false_labels) > 1:
            if gain > 0:
                self.fit(best_false_features, best_false_labels,
                         next_decision_node, decision)

    def predict(self, features):
        '''
        Given some new features, predicts the labels of
        these data points.

        Arguments:
            features: the features of the data set.

        Returns:
            predictions: the predicted labels of the data set.
        '''
        predictions = []

        # For each type of feature in the features.
        for i, feature in enumerate(features):

            # Start from the root of the decision tree.
            node = self.root

            # While there is still a decision to make.
            while node is not None:

                # Ask the question at this node    
                feature_index = node.question[0]
                feat = node.question[1]

                if isinstance(feat, str):
                    condition = feature[feature_index] == feat

                else:
                    condition = feature[feature_index] >= feat


                # If the decision resulted in a true label
                if condition:

                    # If there is no new true decision to make through this node
                    if node.true is None:

                        # All the labels that were classified here in training.
                        possible_labels = node.true_labels
                        number_of_points = len(possible_labels)

                        # The set of labels that were classified here in training.
                        classes = set(possible_labels)
                        class_dict = {}

                        for c in classes:
                            class_dict[c] = 0

                        # Calculate probabilities of each label.
                        for label in possible_labels:
                            class_dict[label] += 1 / number_of_points

                        probabilities = list(class_dict.values())

                        # Predict a label based on frequency of labels in this decision node.
                        predict = np.random.choice(list(classes), p=probabilities)
                        predictions.append(predict)

                        break

                    # Set the next node to the true decision node.
                    node = node.true

                # If the decision resulted in a false label
                else:

                    # If there is no new false decision to make through this node
                    if node.false is None:

                        # All the labels that were classified here in training.
                        possible_labels = node.false_labels
                        number_of_points = len(possible_labels)

                        # The set of labels that were classified here in training.
                        classes = set(possible_labels)
                        class_dict = {}

                        for c in classes:
                            class_dict[c] = 0

                        # Calculate probabilities of each label.
                        for label in possible_labels:
                            class_dict[label] += 1 / number_of_points

                        probabilities = list(class_dict.values())

                           # Predict a label based on frequency of labels in this decision node.
                        predict = np.random.choice(list(classes), p=probabilities)
                        predictions.append(predict)

                        break

                    # Set the next node to the false decision node.
                    node = node.false

        return predictions

    def accuracy(self, true, predicted):
        '''
        Given the true labels of a dataset and the predicted 
        labels of the dataset, calculates the accuracy of the model.

        Arguments:
            true: the true labels of the dataset
            predicted: the predicted labels of the dataset.

        Returns:
            accuracy: the accuracy of the model.
        '''
        correct = 0
        for i in range(len(true)):
            if true[i] == predicted[i]:
                correct += 1

        accuracy = correct / len(true)
        return accuracy

