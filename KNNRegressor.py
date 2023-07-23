import numpy as np
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


class KNNRegressor:
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            dist = self._distance_calc(self.X_train, x)
            closest_args = np.argsort(dist)[: self.n_neighbors]
            y_pred.append(np.mean(self.y_train[closest_args]))

        return y_pred

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)

        tss = np.sum((y_test - np.mean(X_test)) ** 2)
        rss = np.sum((y_test - y_pred) ** 2)

        return 1 - rss/tss

    def _distance_calc(self, X, point):
        return np.linalg.norm(point - X, axis=1, ord=self.p)


# Generating Synthetic Data

X, y = mglearn.datasets.make_wave(n_samples=400)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


knn_reg_own = KNNRegressor()

knn_reg_own.fit(X_train, y_train)

knn_reg = KNeighborsRegressor()

knn_reg.fit(X_train, y_train)


print('Own Model Score: %.2f' % knn_reg_own.score(X_test, y_test))
print('Sklearn Model Score: %.2f' % knn_reg.score(X_test, y_test))
