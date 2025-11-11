# Digit Classification using Nearest Neighbor and k-NN  
This repository contains a clean, modular implementation of **Nearest Neighbor (1-NN)** and **k-Nearest Neighbor (k-NN)** classifiers for handwritten digit classification using the `sklearn` digits dataset.

This project is part of an academic assignment and demonstrates:
- Instance-based learning  
- Distance-based classification  
- Effects of varying k on accuracy  
- Simple, interpretable machine learning without training

The dataset consists of **8×8 grayscale digit images** (64 pixels), with approximately **1800 samples across 10 classes (0–9)**.

---

## Project Structure

digit_knn_classifier/
│
├── classifiers/
│ ├── nn_classifier.py # Nearest Neighbor (k=1)
│ └── knn_classifier.py # k-NN for k=3,5,7
│
├── utils/
│ └── dataset_digits.py # loads dataset and applies class-balanced split
│
├── scripts/
│ └── run_experiments.py # runs all experiments and prints accuracies
│
├── results/
│ ├── accuracies.txt # saved accuracy metrics
│ └── confusion_matrix.png # optional visualization (if enabled)
│
└── README.md # this file

---

## Dataset Description

The dataset is loaded from:

```python
from sklearn.datasets import load_digits
```
Each sample:
Shape: (64,) flattened 8×8 grayscale image
Pixel values range from 0 to 16
Labels: 0–9
This project performs a class-balanced split:
50 test samples per class → 500 test images
Remaining samples used for training
This ensures fair evaluation across all digit classes.
## Implemented Models
1. Nearest Neighbor (1-NN)
Stores training data
Predicts by finding the single closest sample (L2 distance)
2. k-Nearest Neighbor (k = 3, 5, 7)
Computes L2 distances
Selects k smallest distances
Performs unweighted majority vote among labels
No learning or training occurs — these are non-parametric, instance-based classifiers.
## How to Run Experiments
Install dependencies:
```bash
pip install numpy scikit-learn matplotlib seaborn
```
Run the experiment script:
```bash
python scripts/run_experiments.py
```
This prints classification accuracies and saves them in results/accuracies.txt.

## Results
Using your random class-balanced split, the accuracies are:
- 1-NN Accuracy: 0.9780
- 3-NN Accuracy: 0.9780
- 5-NN Accuracy: 0.9760
- 7-NN Accuracy: 0.9760

## Observations:
- k = 1 and k = 3 perform best
- Accuracy decreases slightly for k = 5 and k = 7
- Larger k adds smoothing → may include neighbors from other classes
- High performance overall due to simple, structured dataset

## Confusion Matrix
Enabled inside run_experiments.py, the repository generates -> results/confusion_matrix.png

## Key Concepts Demonstrated
- Instance-based learning
- L2 distance computation
- Majority voting
- Class-balanced test splitting
- k-NN behavior as k increases
- No training (lazy learning)