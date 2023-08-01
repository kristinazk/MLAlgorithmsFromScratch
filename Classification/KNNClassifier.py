import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    """

    This is a class which contains the implementation of the KNN Classifier.

    Parameters:
    n_neighbors - number of closest neighbors values of which we should consider (default = 5)
    p - which distance metric to use (p=1 Manhattan distance, p=2 Euclidean distance etc.) (default = 2)

    Methods:
    fit(X_train, y_train) - used to fit the train data to the model
    predict(X_test) - used to make predictions on test data
    score(X_test, y_test) - used to calculate the score of the model

    Private Methods:
    _distance_calc - used to calculate the distance between two vectors (or a vector and a matrix) using the p value
    specified during class initialisation

    """

    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        y_pred = []
        for x in X:
            dist = self._distance_calc(self.X_train, x)
            closest_args = np.argsort(dist)[: self.n_neighbors]

            unique_values, counts = np.unique(self.y_train[closest_args], return_counts=True)

            y_pred.append(unique_values[np.argmax(counts)])

        return np.array(y_pred)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        return np.sum(y_pred == y_test) / y_test.shape[0]

    def _distance_calc(self, X, point):
        return np.linalg.norm(point - X, ord=self.p, axis=1)


iris = load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn_class_own = KNNClassifier()

knn_class_own.fit(X_train, y_train)

knn_class = KNeighborsClassifier()

knn_class.fit(X_train, y_train)


print('Own Model Score: %.2f' % knn_class_own.score(X_test, y_test))
print('Sklearn Model Score: %.2f' % knn_class.score(X_test, y_test))

