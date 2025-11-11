import numpy as np
from sklearn.datasets import load_digits

def load_digit_dataset():
    """
    Loads digit dataset and creates train/test splits with:
    - 500 test samples (50 per digit class)
    - remaining samples for training
    """
    digits = load_digits()
    X = digits.data.astype(np.float32)       # shape (n_samples, 64)
    y = digits.target

    # normalize from [0,16] → [0,1]
    X /= 16.0

    X_train, y_train = [], []
    X_test, y_test = [], []

    for digit in range(10):
        idx = np.where(y == digit)[0]
        np.random.shuffle(idx)

        test_idx = idx[:50]
        train_idx = idx[50:]

        X_test.append(X[test_idx])
        y_test.append(y[test_idx])

        X_train.append(X[train_idx])
        y_train.append(y[train_idx])

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)

    return X_train, y_train, X_test, y_test
