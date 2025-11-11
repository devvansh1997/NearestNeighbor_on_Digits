import numpy as np

class NearestNeighbor:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        preds = []
        for x in X_test:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argmin(dists)
            preds.append(self.y_train[idx])
        return np.array(preds)
