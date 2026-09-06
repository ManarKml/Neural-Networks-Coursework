# Generative Models and Attention Mechanisms

This directory contains advanced implementations focusing on low-data stabilization using generative models and the integration of spatial attention mechanisms into Convolutional Neural Networks.

**Project Modules**

* **VAE Synthetic Data Generation (`vae-synthetic-data/`):** Trains a Conditional Variational Autoencoder (VAE) on a restricted dataset of only 350 real examples per digit, heavily utilizing initial data augmentation. It generates synthetic examples and filters them into high-confidence ($\ge 0.9$) and mid-confidence ($0.6 \le \text{confidence} \le 0.9$) sets using a baseline LeNet-5 model, then evaluates how these synthetic datasets improve downstream LeNet-5 classification accuracy.
* **Impact of Attention Mechanisms (`attention-mechanisms/`):** Analyzes the performance differences between standard CNNs and attention-augmented CNNs. 
  * **Visual Task:** Compares a standard CNN (like LeNet-5) against a version equipped with spatial attention on the ReducedMNIST dataset to track accuracy and training time.
  * **Speech Task:** Re-evaluates the spectrogram spoken digit recognition task by introducing a spatial attention CNN and comparing it against the baseline model, isolating how attention affects audio feature extraction.

**Dataset Requirements**

* The VAE pipeline specifically restricts the training set to 350 real examples per digit from the ReducedMNIST dataset.
* The attention pipelines require both the full [ReducedMNIST](https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist) dataset and the spoken digits spectrogram dataset.
