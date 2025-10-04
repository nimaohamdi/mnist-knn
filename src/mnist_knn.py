#!/usr/bin/env python3
"""
Handwritten Digit Recognition with KNN (MNIST)
Author: Nimo 
GitHub: https://github.com/nimaohamdi

This script trains a K-Nearest Neighbors classifier on the MNIST dataset.
It provides accuracy evaluation, confusion matrix visualization,
and displays some sample predictions.

Run:
    python mnist_knn.py --k 3 --max-sample 10000

Options:
- Change k to adjust neighbors
- Limit samples for faster training
- Save confusion matrix and sample predictions as images
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# -------------------- Utility Functions --------------------
def plot_confusion_matrix(y_true, y_pred, k, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"KNN (k={k}) - MNIST Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_sample_predictions(X_test, y_test, y_pred, n=10, save_path=None):
    fig, axes = plt.subplots(1, n, figsize=(2*n, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"T:{y_test[i]}\nP:{y_pred[i]}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Sample predictions saved to {save_path}")
    else:
        plt.show()
    plt.close()

# -------------------- Main Script --------------------
def main(k=3, test_size=0.2, max_sample=None, output_dir="results"):
    Path(output_dir).mkdir(exist_ok=True)

    # 1. Load MNIST dataset
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target'].astype(int)

    # Normalize (0-255 → 0-1)
    X = X / 255.0

    # Optionally reduce dataset size for speed
    if max_sample:
        X, y = X[:max_sample], y[:max_sample]
        print(f"Using a subset of {max_sample} samples for faster training.")

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 3. Build KNN model
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    print(f"Training KNN with k={k}...")
    knn.fit(X_train, y_train)

    # 4. Predictions
    y_pred = knn.predict(X_test)

    # 5. Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy with K={k}: {acc:.4f}")

    # 6. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, k, save_path=f"{output_dir}/confusion_matrix.png")

    # 7. Show some predictions
    plot_sample_predictions(X_test, y_test, y_pred, n=8, save_path=f"{output_dir}/sample_predictions.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognition with KNN (MNIST)")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--max-sample", type=int, default=None, help="Limit number of samples for faster training")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save outputs")
    args = parser.parse_args()

    main(k=args.k, test_size=args.test_size, max_sample=args.max_sample, output_dir=args.output_dir)
