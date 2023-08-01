import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class BinaryLogisticRegression:
    """

    This is a class which contains the implementation of Binary Logistic Regression.

    Parameters:
    num_iter - number of iterations to update the model parameters (default = 15000)
    lr - the learning rate to multiply gradients with (default = 0.01)
    threshold - the probability value which splits the classes into positive and negative

    Methods:
    fit(X_train, y_train) - used to fit the train data to the model
    predict(X_test) - used to make predictions on test data
    score(X_test, y_test) - used to calculate the score of the model

    Private Methods:
    _sigmoid(x) - the implementation of the sigmoid function (using a trick to avoid the overflow warning)

    """

    def __init__(self, num_iter=10000, lr=0.001, threshold=0.5):
        self.num_iter = num_iter
        self.lr = lr
        self.threshold = threshold

    def fit(self, X_train, y_train):
        self.bias = 0
        self.weights = np.zeros(X_train.shape[1])

        for _ in range(self.num_iter):
            y_pred = self.predict(X_train)

            db = (1 / X_train.shape[0]) * np.sum(y_pred - y_train)
            dw = (1 / X_train.shape[0]) * np.dot(X_train.T, (y_pred - y_train))

            self.bias -= self.lr * db
            self.weights -= self.lr * dw

    def predict(self, X_train):
        probs = self._sigmoid(np.dot(X_train, self.weights))

        return probs >= self.threshold

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.sum(y_pred == y_test) / len(y_test)

    @staticmethod
    def _sigmoid(x):
        # Use the log-sum-exp trick for numerical stability
        exp_neg_x = np.exp(-np.abs(x))
        return np.where(x >= 0, 1.0 / (1.0 + exp_neg_x), exp_neg_x / (1.0 + exp_neg_x))


X, y = load_breast_cancer().data, load_breast_cancer().target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_reg_own = BinaryLogisticRegression()

log_reg_sklearn = LogisticRegression(max_iter=10000)

log_reg_own.fit(X_train, y_train)
log_reg_sklearn.fit(X_train, y_train)

print('Own Model Score:', log_reg_own.score(X_test, y_test))
print('Sklearn Model Score:', log_reg_sklearn.score(X_test, y_test))
