# MNIST KNN

**Handwritten Digit Recognition using K-Nearest Neighbors (KNN) on the MNIST dataset**

## Features

* Implements the **KNN algorithm** with configurable `k`
* Displays and saves the **confusion matrix**
* Displays and saves **sample predictions**
* Supports limiting the number of samples for faster execution
* Ready for **GitHub** with a proper `.gitignore`

## Project Structure

```
mnist-knn/
├─ README.md          # Project documentation
├─ requirements.txt   # Python dependencies
├─ .gitignore         # Git ignore file
├─ src/
│  └─ mnist_knn.py    # Main project code
├─ notebooks/         # Optional: Jupyter analysis notebooks
└─ results/           # Stores output images and models
```

## Prerequisites

* Python 3.x
* numpy
* matplotlib
* scikit-learn
* joblib

### Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
# Run the model with k=3 on 10,000 samples
python src/mnist_knn.py --k 3 --max-sample 10000 --output-dir results
```

* Outputs, including **confusion matrix** and **sample predictions**, are saved in the `results/` folder.

## Sample Results

* Accuracy is around 96% with k=3 on 10,000 samples
* Confusion matrix and sample predictions are saved automatically

