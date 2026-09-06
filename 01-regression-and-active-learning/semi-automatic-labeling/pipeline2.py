import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate, shift
from sklearn.svm import SVC
from check_accuracy import check_accuracy
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
np.random.seed(0)

# STAGE 1: DATA LOADING & HOG EXTRACTION

def load_raw_images(folder_path):
    images = []
    for i in range(1, 10001):
        img_path = os.path.join(folder_path, f"{i}.bmp")
        img = Image.open(img_path).convert('L')
        # Normalize to [0, 1]         
        images.append(np.array(img).flatten() / 255.0)
    return np.array(images)

def get_hog_features(raw_data_array):
    hog_list = []
    for img_flat in raw_data_array:
        img = img_flat.reshape(28, 28)
        # HOG parameters (orientations, pixels_per_cell, cells_per_block) 
        feat = hog(img, orientations=9, pixels_per_cell=(7, 7), 
                   cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_list.append(feat)
    return np.array(hog_list)

# Load the raw pixels
X_raw_all = load_raw_images('F:/EECE 26/EECE - Year 4/_2nd_term/Neural Networks/Assignments/Assignment 01/Part3/Indian_Digits_Train')

# Extract HOG features for the entire dataset 
X_hog_all = get_hog_features(X_raw_all)
scaler = StandardScaler()
X_all = scaler.fit_transform(X_hog_all)

# STAGE 2: MODE SELECTION & SEED INITIALIZATION

print("Select Pipeline 2 Execution Mode:")
print("[1] Automated Run (Uses known_labels dictionary for refinements)")
print("[2] Interactive Training (User must manually label refinements)")
choice = input("Enter choice (1 or 2): ")
auto_mode = True if choice == '1' else False

# Load initial 300 seed labels  
csv_data = pd.read_csv('F:/EECE 26/EECE - Year 4/_2nd_term/Neural Networks/Assignments/Assignment 01/Part3/my_labels.csv')
known_labels = dict(zip(csv_data['image'], csv_data['label']))

# 60 labels from previous runs to automate the refinements stage to see the results directly
my_refinements = {
    # Iteration 1
    4000: 5, 3971: 5, 6934: 6, 4522: 5, 373: 5, 1678: 1, 3554: 5, 2145: 5, 
    5470: 7, 1217: 4, 1335: 5, 5979: 9, 6248: 5, 2867: 6, 9946: 5, 3283: 1, 
    5712: 7, 9790: 3, 8740: 5, 9613: 2,
    # Iteration 2
    3823: 7, 439: 3, 7037: 2, 8242: 9, 4859: 2, 965: 3, 6362: 9, 7183: 4, 
    6844: 2, 2343: 9, 8644: 2, 2141: 5, 9498: 4, 9295: 5, 740: 9, 4964: 9, 
    1212: 5, 1672: 2, 5733: 7, 9037: 5,
    # Iteration 3
    6914: 2, 3892: 6, 8579: 8, 9254: 2, 3265: 2, 4088: 2, 9369: 2, 9144: 5, 
    4295: 9, 816: 6, 629: 3, 6128: 7, 9211: 2, 8008: 3, 436: 9, 4778: 1, 
    8219: 6, 5392: 7, 884: 7, 2476: 2
}
known_labels.update(my_refinements)

seed_indices = csv_data['image'].values - 1
y_seed = csv_data['label'].values

# Initial Training Pool (Weight 200 for trusted human labels) 
X_train = X_all[seed_indices]
y_train = y_seed
train_weights = np.full(len(y_seed), 200) 

total_manual_time = len(y_seed) * 10 
unlabeled_mask = np.ones(10000, dtype=bool)
unlabeled_mask[seed_indices] = False

# STAGE 3: DATA AUGMENTATION 

def augment_seed_set(X_raw_samples, y_samples):
    X_aug_raw, y_aug = [], []
    for img_flat, label in zip(X_raw_samples, y_samples):
        img = img_flat.reshape(28, 28)
        
        # 1. Rotations +/- 5 deg 
        X_aug_raw.append(rotate(img, 5, reshape=False).flatten())
        X_aug_raw.append(rotate(img, -5, reshape=False).flatten())
        
        # 2. Spatial shifts 
        for s in [(1,0), (-1,0), (0,1), (0,-1)]:
            X_aug_raw.append(shift(img, s).flatten())
            
        # 3. Gaussian Noise 
        noise = np.random.normal(0, 0.05, img.shape)
        noisy_img = np.clip(img + noise, 0, 1)
        X_aug_raw.append(noisy_img.flatten())
        
        # Total of 7 augmented copies per seed image
        for _ in range(7): 
            y_aug.append(label) 
            
    X_aug_hog = get_hog_features(np.array(X_aug_raw))
    return X_aug_hog, np.array(y_aug)

# Process augmentation on raw data, then convert to HOG
X_seed_raw = X_raw_all[seed_indices]
X_aug_hog, y_aug = augment_seed_set(X_seed_raw, y_seed)
X_aug_hog_scaled = scaler.transform(X_aug_hog)

# Add augmented data to training pool (Weight 1) 
X_train = np.vstack([X_train, X_aug_hog_scaled])
y_train = np.concatenate([y_train, y_aug])
train_weights = np.concatenate([train_weights, np.full(len(y_aug), 1)])

# STAGE 4: ITERATIVE ACTIVE & SELF TRAINING

# Multi-class SVM with RBF Kernel  
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)

iteration = 1
accuracy = 0.0
while accuracy < 0.99:
    print(f"\n--- Iteration {iteration} ---")
    svm_model.fit(X_train, y_train, sample_weight=train_weights) 
    
    # 1. Calculate Confidence Margins using Probabilities 
    unlabeled_indices = np.where(unlabeled_mask)[0]
    probs = svm_model.predict_proba(X_all[unlabeled_indices])
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2] 
    
    # 2. Active Refinement: 20 Smallest Margins 
    uncertain_local_idx = np.argsort(margins)[:20]
    uncertain_global_idx = unlabeled_indices[uncertain_local_idx]
    
    for idx in uncertain_global_idx:
        image_number = idx + 1
        if auto_mode and image_number in known_labels:
            label = known_labels[image_number]
        else:
            plt.imshow(X_raw_all[idx].reshape(28,28), cmap='gray')
            plt.title(f"Identify Image {image_number}.bmp")
            plt.draw()
            plt.pause(0.001)
            label = int(input(f"Enter label for image {image_number}: "))
            plt.close()
            known_labels[image_number] = label 
            
        X_train = np.vstack([X_train, X_all[idx]])
        y_train = np.append(y_train, label)
        train_weights = np.append(train_weights, 200) # Human weight 
        unlabeled_mask[idx] = False
        total_manual_time += 10 

    # 3. Self-Training: Incorporation of High-Confidence Predictions
    threshold = np.percentile(margins, 90) # Higher threshold to ensure accuracy 
    preds = svm_model.predict(X_all[unlabeled_indices])
    
    for class_id in range(10):
        class_mask = (preds == class_id) & (margins > threshold) 
        class_pool = unlabeled_indices[class_mask]
        
        top_30 = class_pool[np.argsort(margins[class_mask])[-30:]]
        if len(top_30) > 0:
            X_train = np.vstack([X_train, X_all[top_30]])
            y_train = np.concatenate([y_train, preds[np.isin(unlabeled_indices, top_30)]]) 
            train_weights = np.concatenate([train_weights, np.full(len(top_30), 1)]) 
            unlabeled_mask[top_30] = False
            
    # 4. Evaluation 
    final_preds = svm_model.predict(X_all)
    accuracy, _, _ = check_accuracy(final_preds) 
    print(f"Accuracy: {accuracy*100:.2f}% | Manual Time: {total_manual_time/60:.2f} min") 
    iteration += 1


print("\n--- Pipeline 2 Complete ---")
print(f"Final Labelling Accuracy: {accuracy*100:.2f}%") 
print(f"Total Iterations: {iteration-1}") 
print(f"Total Manual Time: {total_manual_time/60:.2f} minutes") 