# Regression and Active Learning Pipelines

This directory contains implementations for polynomial regression modeling and semi-automatic data labeling workflows. 

**Project Modules**

* **Insurance Data Regression (`insurance_regression.py`):** Fits linear, quadratic, and cubic functions to predict the number of insured persons over time. It evaluates model performance using R-squared values and compares results before and after removing data outliers.
* **Pipeline 1: K-Means Bootstrapping & Active SVM (`pipeline1_kmeans_svm.py`):** Combines unsupervised clustering with an iterative, human-in-the-loop SVM utilizing an RBF kernel. It labels entire clusters initially, then focuses manual effort on ambiguous images near the decision boundary to reach a 99% accuracy target.
* **Pipeline 2: Manual Seed Labeling & Self-Training (`pipeline2_selftraining.py`):** Begins with 300 manually labeled seed images and expands the dataset through spatial augmentation, including rotation, shifting, and noise. It iteratively adds high-confidence pseudo-labeled images back into the training set.

**Dataset Requirements**

* The active learning pipelines require the [Indian_Digits_Train](https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist) dataset, which consists of 10,000 unlabelled 28x28 grayscale images.
* Ensure the dataset folder is located in the root of this directory before executing the scripts.

**Execution**

Run the scripts directly via the command line:
`python insurance_regression.py`
or
`python pipeline1_kmeans_svm.py`
or
`python pipeline2_selftraining.py`
