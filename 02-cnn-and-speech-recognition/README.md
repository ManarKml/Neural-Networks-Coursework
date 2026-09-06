# CNNs and Speech Recognition

This directory contains Convolutional Neural Network (CNN) architectures implemented for both visual digit classification and spectrogram-based speech recognition. 

**Project Modules**

* **LeNet-5 Image Classification (`lenet5-reduced-mnist/`):** Implements a modified LeNet-5 architecture adapted for 28x28 ReducedMNIST images, utilizing ReLU activation functions instead of the standard sigmoid/tanh. The module includes scripts to evaluate different hyperparameter variations, such as altering the number of filters in convolutional layers or modifying activation functions, and logs the resulting training times, testing times, and accuracies.
* **Spectrogram Speech Recognition (`spectrogram-speech-recognition/`):** Treats audio classification as an object recognition task by training a CNN on spectrogram images of spoken digits. The pipeline evaluates model robustness under four specific conditions:
  * Baseline training on raw spectrograms.
  * Speech data augmentation (altering audio speed by up/down 3% and injecting speech noise).
  * Image data augmentation (horizontally squeezing or expanding the spectrogram images by 3% and adding noise).
  * A combined augmentation approach utilizing both audio and visual modifications.

**Dataset Requirements**

* The visual classification pipeline requires the [Indian_Digits_Train](https://www.kaggle.com/datasets/mohamedgamal07/reduced-mnist) dataset (1000 training examples and 200 test examples per digit).
* The speech recognition pipeline requires the spoken digits audio dataset. 
