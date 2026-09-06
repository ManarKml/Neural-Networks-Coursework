import os
import time
import librosa
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. CONFIGURATION ---

IMG_SIZE = (64, 64)
train_path = 'F:/Neural Networks/Assignment 2/audio-dataset/Train'
test_path = 'F:/Neural Networks/Assignment 2/audio-dataset/Test'

# --- 2. DATA UTILITIES ---

def extract_label(filename):
    """Extracts the digit after the underscore: 'C04_6.wav' -> 6"""
    return int(filename.split('_')[-1].split('.')[0])

def normalize(data):
    """Standardizes spectrogram values to 0-1 range"""
    return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

def get_audio_spectrogram(y, sr):
    """Converts raw audio to a normalized 64x64 spectrogram image"""
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    resized = cv2.resize(S_db, IMG_SIZE)
    return normalize(resized)

def load_audio_dataset(folder):
    """Reused loader to ensure fair comparison with Assignment 2"""
    X, y = [], []
    for file in os.listdir(folder):
        if not file.endswith('.wav'): continue
        path = os.path.join(folder, file)
        audio, sr = librosa.load(path, sr=None) 
        X.append(get_audio_spectrogram(audio, sr))
        y.append(extract_label(file))
    return np.array(X).reshape(-1, 64, 64, 1), np.array(y)

def load_reduced_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    indices = []
    for i in range(10):
        idx = np.where(y_train == i)[0][:350] 
        indices.extend(idx)
    return x_train[indices], y_train[indices], x_test, y_test

# --- 3. SPATIAL ATTENTION & MODEL BUILDERS ---

def spatial_attention(input_feature, kernel_size=7):
    """
    Spatial attention doesn't care which channel contains the feature/edge 
    It cares where (coordinates) in the image that feature/edge was found
    So we implement avg & max pooling across all channels (pixel 1 channel 1 with pixel 1 channel 2...) 
    Then concat the outputs into 2 channels and do conv between them 
    The output is a filter/mask that contains an importance score for each pixel
    """
    # perform avg and max pooling 
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    
    # Concatenate the pooled maps
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Generate the attention mask
    attention = layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same', activation='sigmoid', use_bias=False)(concat)
    
    # Multiply the original features by the mask
    return layers.Multiply()([input_feature, attention])

def build_part_a_lenet(input_shape=(28, 28, 1), use_attention=False):
    """
    LeNet-5 for Part (a)
    Uses 5x5 filters and smaller filters (6, 16)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1: 6 filters, 5x5
    x = layers.Conv2D(6, (5, 5), activation='relu', padding='same')(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    # Layer 2: 16 filters, 5x5
    x = layers.Conv2D(16, (5, 5), activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    if use_attention:
        x = spatial_attention(x)
        
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_part_b_speech_model(input_shape=(64, 64, 1), use_attention=False):
    """
    Assignment 2 Model for Part (b):
    Uses 3x3 filters and larger filters (32, 64)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Layer 1: 32 filters, 3x3
    x = layers.Conv2D(32, (3, 3), activation='relu', padding="same")(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    # Layer 2: 64 filters, 3x3
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    
    if use_attention:
        x = spatial_attention(x)
        
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# --- 4. EXECUTION ---

# Part (a) Execution
print("--- Running Part (a) MNIST ---")
x_train_m, y_train_m, x_test_m, y_test_m = load_reduced_mnist()

results_mnist = {}
mnist_models = {}

for use_attn in [False, True]:
    mode = "Attention" if use_attn else "Baseline"
    model = build_part_a_lenet(use_attention=use_attn)
    start = time.time()
    model.fit(x_train_m, y_train_m, epochs=10, batch_size=32, verbose=0)
    train_time = time.time() - start
    acc = model.evaluate(x_test_m, y_test_m, verbose=0)[1]
    results_mnist[mode] = (acc, train_time)
    mnist_models[mode] = model # Store model for CM
    print(f"{mode} MNIST: Accuracy = {acc*100:.2f}%, Time = {train_time:.1f}s")

# Part (b) Execution
print("\n--- Running Part (b) Spoken Digits ---")
X_train, y_train = load_audio_dataset(train_path)
X_test, y_test = load_audio_dataset(test_path)

results_speech = {}
speech_models = {}

for use_attn in [False, True]:
    mode = "Attention" if use_attn else "Baseline"    
    model = build_part_b_speech_model(use_attention=use_attn)
    start_time = time.time()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    train_time = time.time() - start_time
    acc = model.evaluate(X_test, y_test, verbose=0)[1]
    results_speech[mode] = (acc, train_time)
    speech_models[mode] = model # Store model for CM
    print(f"{mode} Spoken Digits: Accuracy = {acc*100:.2f}%, Time = {train_time:.1f}s")

# --- 5. VISUALIZATION ---

def plot_comparison_charts(results_mnist, results_speech):
    """Draws bar charts for Accuracy and Training Time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Comparison
    labels = ['MNIST Base', 'MNIST Attn', 'Speech Base', 'Speech Attn']
    accs = [results_mnist['Baseline'][0]*100, results_mnist['Attention'][0]*100, 
            results_speech['Baseline'][0]*100, results_speech['Attention'][0]*100]
    
    axes[0].bar(labels, accs, color=['blue', 'darkblue', 'green', 'darkgreen'])
    axes[0].set_title('Accuracy Comparison (%)')
    axes[0].set_ylim(min(accs) - 2, 100)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha='center')

    # Time Comparison
    times = [results_mnist['Baseline'][1], results_mnist['Attention'][1], 
             results_speech['Baseline'][1], results_speech['Attention'][1]]
    
    axes[1].bar(labels, times, color=['skyblue', 'blue', 'lightgreen', 'green'])
    axes[1].set_title('Training Time Comparison (s)')
    for i, v in enumerate(times):
        axes[1].text(i, v + 0.1, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(model_base, model_attn, X_test, y_test, title_prefix):
    """Plots side-by-side confusion matrices."""
    y_pred_base = np.argmax(model_base.predict(X_test, verbose=0), axis=1)
    y_pred_attn = np.argmax(model_attn.predict(X_test, verbose=0), axis=1)
    
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_attn = confusion_matrix(y_test, y_pred_attn)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'{title_prefix} Confusion Matrices', fontsize=14, fontweight='bold')

    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'{title_prefix} Baseline CM')
    
    sns.heatmap(cm_attn, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'{title_prefix} Attention CM')
    
    plt.show()

def plot_attention_overlay(model_base, model_attn, X_test, y_test, num_samples=3):
    """Extracts the internal attention mask and overlays it on the original image."""
# 1. Find the attention layer
    attn_layer = None
    for layer in model_attn.layers:
        if isinstance(layer, layers.Conv2D) and layer.activation.__name__ == 'sigmoid':
            attn_layer = layer
            break
            
    if attn_layer is None:
        print("Could not find attention layer.")
        return

    # 2. Create a sub-model to extract the internal mask
    mask_model = models.Model(inputs=model_attn.input, outputs=attn_layer.output)
    
    # 3. Pick random test samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    samples = X_test[indices]
    labels = y_test[indices]
    
    # 4. Get Predictions & Masks
    masks = mask_model.predict(samples, verbose=0)
    preds_attn = np.argmax(model_attn.predict(samples, verbose=0), axis=1)
    preds_base = np.argmax(model_base.predict(samples, verbose=0), axis=1)
    
    # 5. Plotting
    fig, axes = plt.subplots(num_samples, 2, figsize=(7, 3.5 * num_samples))
    
    for i in range(num_samples):
        img = samples[i].squeeze()
        mask = masks[i].squeeze()
        
        # Upscale the tiny 7x7 mask back to 28x28
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        # Multiply the image by the mask. 
        attended_img = img * mask_resized
        
        # Determine axes for this row
        ax1 = axes[i, 0] if num_samples > 1 else axes[0]
        ax2 = axes[i, 1] if num_samples > 1 else axes[1]
        
        # Left Column: Original Digit
        ax1.imshow(img, cmap='gray')
        ax1.set_title("Original Digit")
        ax1.axis('off')
        
        # Right Column: Attention Overlay
        ax2.imshow(attended_img, cmap='jet')
        ax2.set_title("Attention Overlay")
        ax2.axis('off')
        
    plt.tight_layout()
    plt.show()

# --- GENERATE ALL PLOTS ---
plot_comparison_charts(results_mnist, results_speech)
plot_confusion_matrices(mnist_models['Baseline'], mnist_models['Attention'], x_test_m, y_test_m, "MNIST")
plot_confusion_matrices(speech_models['Baseline'], speech_models['Attention'], X_test, y_test, "Speech")

# Run the Attention Overlay visualization
plot_attention_overlay(mnist_models['Baseline'], mnist_models['Attention'], x_test_m, y_test_m, num_samples=3)
plot_attention_overlay(speech_models['Baseline'], speech_models['Attention'], X_test, y_test, num_samples=3)
