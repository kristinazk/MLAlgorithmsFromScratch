import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class LinearRegressionOwn:
    """

       This is a class which contains the implementation of Linear Regression.

       Parameters:
       num_iter - number of iterations to update the model parameters (default = 15000)
       lr - the learning rate to multiply gradients with (default = 0.01)
       regularization - applies regularization to the model to avoid overfitting
       possible values: {None, 'ridge', 'lasso', 'elnet')}, default: None
       alpha - controls the amount of regularization applied to the model (default = 1)


       Methods:
       fit(X_train, y_train) - used to fit the train data to the model
       predict(X_test) - used to make predictions on test data
       score(X_test, y_test) - used to calculate the score of the model

       """
    def __init__(self, num_iter=15000, lr=0.01, regularization=None, alpha=1.0):
        self.num_iter = num_iter
        self.lr = lr
        self.regularization = regularization
        self.alpha = alpha

    def fit(self, X_train, y_train):
        n_features = X_train.shape[1]
        n_elements = X_train.shape[0]

        self.weights = np.random.randn(n_features)
        self.bias = np.random.normal(0, 1)

        y_pred = self.predict(X_train)

        dw = np.zeros(n_features)

        for _ in range(self.num_iter):
            base_deriv = - 2 * X_train.T @ (y_train - y_pred)

            if not self.regularization:
                dw = base_deriv / n_elements

            elif self.regularization.lower() == 'ridge':
                dw = (base_deriv + 2 * self.alpha * self.weights) / n_elements

            elif self.regularization.lower() == 'lasso':
                for i in range(n_features):
                    if self.weights[i] < 0:
                        dw[i] = (X_train.T[i, :] @ (y_train - y_pred) - self.alpha) / n_elements
                    else:
                        dw[i] = (X_train.T[i, :] @ (y_train - y_pred) + self.alpha) / n_elements

            elif self.regularization.lower() == 'elnet':
                for i in range(n_features):
                    if self.weights[i] < 0:
                        dw[i] = (X_train.T[i, :] @ (y_train - y_pred) + 2 * self.alpha * self.weights[i] - self.alpha) / n_elements
                    else:
                        dw[i] = (X_train.T[i, :] @ (y_train - y_pred) + 2 * self.alpha * self.weights[i] + self.alpha) / n_elements

            db = - 2 * np.mean((y_train - y_pred))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        return X_test @ self.weights + self.bias

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        return 1 - np.sum((y_test - np.mean(y_test)) ** 2) / np.sum((y_test - y_pred) ** 2)


X, y, = make_regression(n_samples=200, n_features=7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lreg_own = LinearRegressionOwn(regularization='ridge', alpha=0.1)

lreg_sklearn = LinearRegression()

lreg_own.fit(X_train, y_train)
lreg_sklearn.fit(X_train, y_train)

print("Own model score:", lreg_own.score(X_test, y_test))
print("Sklearn model score:", lreg_sklearn.score(X_test, y_test))
