# Neural Networks & Machine Learning Coursework

This repository contains a collection of Python-based machine learning pipelines and deep neural network architectures developed during university coursework. The projects span classical machine learning regression, semi-automatic active learning, and speech recognition.

**Used Libraries:** Python, PyTorch, TensorFlow, MATLAB, scikit-learn

**Repository Details**

* **`01-regression-and-active-learning/`**
  * `insurance-regression/`: Scripts fitting linear, quadratic, and cubic regression models to dataset trends with outlier removal.
  * `semi-automatic-labeling/`: Active learning pipelines to minimize manual data labeling on a 10,000-image dataset. Includes K-Means bootstrapping with active SVM refinement and a separate pipeline for manual seed labeling with self-training.

* **`02-cnn-and-speech-recognition/`**
  * `lenet5-reduced-mnist/`: Custom LeNet-5 Convolutional Neural Network architectures for $28×28$ image classification, featuring varied hyperparameter tuning and performance benchmarking.
  * `spectrogram-speech-recognition/`: Deep learning model for classifying spoken digits using spectrogram images, tracking performance impacts across various speech and image data augmentation techniques.

* **`03-generative-models-and-attention/`**
  * `vae-data-stabilization/`: A Conditional Variational Autoencoder (VAE) trained to generate synthetic examples from a limited set of 350 real images, aimed at stabilizing low-data training environments.
  * `spatial-attention-cnn/`: Implementation and comparative analysis of standard CNNs versus spatial attention CNNs for both visual and spectrogram-based speech recognition tasks.

**Execution Steps**

1. Clone this repository: `git clone https://github.com/ManarKml/Neural-Networks-Coursework.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Navigate to individual project directories to execute specific `.py` scripts. Each module contains localized comments for dataset requirements.
