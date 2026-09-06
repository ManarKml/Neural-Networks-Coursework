import os
import librosa
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import random

# Set the fixed seed for all relevant libraries
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- 1. CONFIGURATION & UTILS ---
IMG_SIZE = (64, 64)

def extract_label(filename):
    """Extracts the digit after the underscore: 'C04_6.wav' -> 6"""
    return int(filename.split('_')[-1].split('.')[0])

def normalize(data):
    """Standardizes spectrogram values to 0-1 range"""
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

# --- 2. AUGMENTATION FUNCTIONS ---

def get_audio_spectrogram(y, sr):
    """Converts raw audio to a normalized 64x64 spectrogram image"""
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    resized = cv2.resize(S_db, IMG_SIZE)
    return normalize(resized)

def speech_augment(y, sr):
    """Part b: Speed up 3%, down 3%, and add noise"""
    y_fast = librosa.effects.time_stretch(y, rate=1.03)
    y_slow = librosa.effects.time_stretch(y, rate=0.97)
    noise = y + 0.005 * np.random.randn(len(y))
    return [y_fast, y_slow, noise]

def image_augment(spec):
    """Part c: Squeeze/Expand horizontally by 3% and add image noise"""
    h, w = spec.shape
    # Squeeze
    squeezed = cv2.resize(cv2.resize(spec, (int(w*0.97), h)), (w, h))
    # Expand
    expanded = cv2.resize(cv2.resize(spec, (int(w*1.03), h)), (w, h))
    # Noise
    noisy = spec + np.random.randn(h, w)
    return [squeezed, expanded, noisy]

# --- 3. DATA LOADING ENGINE ---

def load_dataset(folder, mode='a'):
    """
    mode 'a': baseline
    mode 'b': speech aug
    mode 'c': image aug
    mode 'd': both
    """
    X, y = [], []
    for file in os.listdir(folder):
        if not file.endswith('.wav'): continue
        
        path = os.path.join(folder, file)
        audio, sr = librosa.load(path, sr=None) 
        label = extract_label(file)
        
        # Original Spectrogram (Always included)
        spec = get_audio_spectrogram(audio, sr)
        X.append(spec)
        y.append(label)
        
        # Part b: Speech Augmentation
        if mode in ['b', 'd']:
            for aug_audio in speech_augment(audio, sr):
                X.append(get_audio_spectrogram(aug_audio, sr))
                y.append(label)
        
        # Part c: Image Augmentation
        if mode in ['c', 'd']:
            for aug_spec in image_augment(spec):
                X.append(aug_spec)
                y.append(label)
                
    return np.array(X).reshape(-1, 64, 64, 1), np.array(y)

# --- 4. MODEL & TRAINING ---

def build_speech_model():
    """CNN based on Problem 2 LeNet-5 structure"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1), padding="same"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 5. EXECUTION ---

modes = ['a', 'b', 'c', 'd']
results = {}

# Pre-load Test Data
X_test, y_test = load_dataset('F:/Neural Networks/Assignment 2/audio-dataset/Test', mode='a')

for m in modes:
    print(f"\n--- Starting Problem 3 Part {m} ---")
    X_train, y_train = load_dataset('F:/Neural Networks/Assignment 2/audio-dataset/Train', mode=m)
    
    model = build_speech_model()
    start_time = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    train_time = time.time() - start_time
    
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    results[m] = (acc, train_time)
    print(f"Part {m} Accuracy: {acc*100:.1f}%, Training Time: {train_time:.1f}s")

# --- 6. VISUALIZTION ---

def get_spec(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    return cv2.resize(S_db, (64, 64))

# Pick one sample file from the Train folder
sample_path = 'F:/Neural Networks/Assignment 2/audio-dataset/Train/C03_0.wav'
y, sr = librosa.load(sample_path, sr=None)

# Generate Versions
orig = get_spec(y, sr)
# Speech Aug (Part b)
y_fast = librosa.effects.time_stretch(y, rate=1.03)
y_slow = librosa.effects.time_stretch(y, rate=0.97)
y_noise = y + 0.005 * np.random.randn(len(y))
# Image Aug (Part c)
h, w = orig.shape
squeezed = cv2.resize(cv2.resize(orig, (int(w*0.97), h)), (w, h))
expanded = cv2.resize(cv2.resize(orig, (int(w*1.03), h)), (w, h))
img_noise = orig + np.random.randn(h, w)

# Plotting
titles = ['Original', 'Speed Up', 'Slow Down', 'Audio Noise', 'Squeezed', 'Expanded', 'Image Noise']
images = [orig, get_spec(y_fast, sr), get_spec(y_slow, sr), get_spec(y_noise, sr), squeezed, expanded, img_noise]

plt.figure(figsize=(18, 3)) 
for i in range(len(images)):
    plt.subplot(1, 7, i+1) 
    plt.imshow(images[i], aspect='equal', origin='lower') 
    plt.title(titles[i], fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# should the noisy audio/images in part b and c be different from the augmented ones or added to them?