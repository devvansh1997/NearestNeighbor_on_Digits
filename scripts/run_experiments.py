import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from utils.dataset_digits import load_digit_dataset
from models.nn_classifier import NearestNeighbor
from models.knn_classifier import KNearestNeighbor

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

X_train, y_train, X_test, y_test = load_digit_dataset()

results = {}

# 1-NN
model_1nn = NearestNeighbor()
model_1nn.fit(X_train, y_train)
pred_1nn = model_1nn.predict(X_test)
results["1-NN"] = accuracy(y_test, pred_1nn)

# k-NN variations
for k in [3, 5, 7]:
    model_knn = KNearestNeighbor(k=k)
    model_knn.fit(X_train, y_train)
    pred = model_knn.predict(X_test)
    results[f"{k}-NN"] = accuracy(y_test, pred)

# print and save results
for k, acc in results.items():
    print(f"{k} Accuracy: {acc:.4f}")

with open("results/accuracies.txt", "w") as f:
    for k, acc in results.items():
        f.write(f"{k}: {acc:.4f}\n")

cm = confusion_matrix(y_test, pred_1nn)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix (1-NN)")
plt.savefig("results/confusion_matrix.png", dpi=300)