import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

class GaussianNaiveBayes:
    """

    This is a class which contains the implementation of the Gaussian Naive Bayes Classifier.


    Methods:
    fit(X_train, y_train) - used to fit the train data to the model
    predict(X_test) - used to make predictions on test data
    score(X_test, y_test) - used to calculate the score of the model
    make_dist(data) - takes the data and generates a Gaussian distribution based on mean and covariance of data.
    class_mapper(classes) - used to map class labels to indices in case they are incompatible (ex. string).

    Private Methods:
    _make_dist(data) - takes the data and generates a Gaussian distribution based on mean and covariance of data
    _class_mapper(classes) - used to map class labels to indices in case they are incompatible (ex. string)

    """
    def __init__(self):
        self.dists = []
        self.priors = []

    def fit(self, X_train, y_train):
        unique_classes = np.sort(np.unique(y_train))
        self.class_mappings = self._class_mapper(unique_classes)

        for cl in unique_classes:
            X_based_y = X_train[y_train == cl]
            self.priors.append(len(X_based_y))
            feature_dist = []
            for feature_idx in range(X_based_y.shape[1]):
                feature_dist.append(self._make_dist(X_based_y[:, feature_idx]))
            self.dists.append(feature_dist)

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            probs = []
            for dist_idx in range(len(self.dists)):
                product = self.priors[dist_idx]
                for feature_idx in range(len(self.dists[0])):
                    product *= self.dists[dist_idx][feature_idx].pdf(x[feature_idx])
                probs.append(product)
            y_pred.append(self.class_mappings[np.argmax(probs)])
        return np.array(y_pred)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test) / len(y_test)

    @staticmethod
    def _make_dist(data):
        mean = np.mean(data)
        cov = np.cov(data)

        return norm(mean, cov)

    @staticmethod
    def _class_mapper(class_names):
        output = {}
        for i, name in enumerate(class_names):
            output[i] = name

        return output


n_bayes_own = GaussianNaiveBayes()

n_bayes_sklearn = GaussianNB()

X, y = make_blobs(n_samples=1000, centers=40, n_features=5, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

n_bayes_own.fit(X_train, y_train)

n_bayes_sklearn.fit(X_train, y_train)

print("Own Model Score", n_bayes_own.score(X_test, y_test))

print("Sklearn Model Score", n_bayes_sklearn.score(X_test, y_test))
