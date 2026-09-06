import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import time
import random 

seed_value = 42

# Set the fixed seed for all relevant libraries
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def build_model(activation='relu', filters_c1=6, ker_size=(5, 5)):
    model = models.Sequential([
        # CONV 1 with variable filters and activation
        layers.Conv2D(filters_c1, ker_size, activation=activation, input_shape=(28, 28, 1)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        # CONV 2
        layers.Conv2D(16, ker_size, activation=activation),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        
        layers.Flatten(),
        layers.Dense(120, activation=activation), # FC 1 
        layers.Dense(84, activation=activation),  # FC 2 
        layers.Dense(10, activation='softmax')    # Output
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Setup Variations for the Table 
# Format: (Activation, C1_Filters, Kernel_Size, Variant)
variations = [
    ('relu', 6, (5, 5), 'none'),         # Baseline
    ('sigmoid', 6, (5, 5), 'none'),      # Var 1: sigmoid as act func
    ('relu', 12, (5, 5), 'none'),        # Var 2: inc CONV1 filter size
    ('relu', 6, (5, 5), 'add_CONV3'),    # Var 3: adding CONV3 layer
    ('relu', 6, (3, 3), 'none')          # Var 4: changing kernel size to 3x3
]

# Variation 3: Adding CONV3 Layer
def build_variation_3():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # added CONV3 layer before FC1
        layers.Flatten(),
        layers.Dense(120, activation='relu'),     # FC2
        layers.Dense(84, activation='relu'),      # FC2 
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare ReducedMNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Logic to reduce dataset: 1000 per digit for training, 200 for testing 
def reduce_dataset(x, y, count_per_digit):
    x_reduced, y_reduced = [], []
    for i in range(10):
        idx = np.where(y == i)[0]
        selected_idx = np.random.choice(idx, count_per_digit, replace=False)
        x_reduced.append(x[selected_idx])
        y_reduced.append(y[selected_idx])
    return np.concatenate(x_reduced), np.concatenate(y_reduced)

x_train_red, y_train_red = reduce_dataset(x_train, y_train, 1000)
x_test_red, y_test_red = reduce_dataset(x_test, y_test, 200)

# Normalize and reshape
x_train_red = x_train_red.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test_red = x_test_red.reshape(-1, 28, 28, 1).astype('float32') / 255.0

for i, (act, filt, ker, var) in enumerate(variations):
    print(f"\n--- Running Variation {i}: {act}, Filters={filt}, Kernel size={ker} ---")
    
    if var == 'add_CONV3':
        model = build_variation_3()
    else:
        model = build_model(activation=act, filters_c1=filt, ker_size=ker)
        
    # Track Training Time 
    start_train = time.time()
    model.fit(x_train_red, y_train_red, epochs=10, batch_size=32)
    end_train = time.time()
    train_time_ms = (end_train - start_train) * 1000    
    
    # Track Testing Time and Accuracy
    start_test = time.time()
    loss_val, acc = model.evaluate(x_test_red, y_test_red)
    end_test = time.time()
    test_time_ms = (end_test - start_test) * 1000

    print(f"Accuracy: {acc * 100:.1f}%") 
    print(f"Training Time: {train_time_ms:.1f} msec")
    print(f"Testing Time: {test_time_ms:.1f} msec")
