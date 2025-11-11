import numpy as np

class KNearestNeighbor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        preds = []
        for x in X_test:
            dists = np.linalg.norm(self.X_train - x, axis=1)
            knn_idx = np.argsort(dists)[:self.k]
            knn_labels = self.y_train[knn_idx]

            # majority vote
            vals, counts = np.unique(knn_labels, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)
