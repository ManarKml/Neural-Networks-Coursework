import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from check_accuracy import check_accuracy
np.random.seed(0)

# Stage 0: STORED DATA
# Labels from previous runs for the automated run mode to see the achieved results

# Cluster labels to automate the clustering stage
my_clusters = {
    0: 2, 1: 6, 2: 4, 3: 0, 4: None, 5: 1, 6: 1, 7: 8, 8: 0, 9: 5,
    10: 8, 11: 3, 12: None, 13: 6, 14: 5, 15: 3, 16: 7, 17: 9, 18: 7, 19: 7,
    20: 0, 21: 2, 22: 8, 23: 5, 24: 9, 25: 4, 26: 2, 27: 9, 28: 1, 29: 0,
    30: 2, 31: 2, 32: 7, 33: 6, 34: 6, 35: 7, 36: 9, 37: 5, 38: 5, 39: 4,
    40: 4, 41: None, 42: 8, 43: 8, 44: 7, 45: 9, 46: 8, 47: 7, 48: 7, 49: 6,
    50: 1, 51: 7, 52: 4, 53: 3, 54: 0, 55: 8, 56: 0, 57: 2, 58: 4, 59: 4,
    60: 2, 61: 5, 62: 5, 63: 5, 64: 6, 65: 3, 66: 7, 67: 4, 68: 1, 69: 3,
    70: 3, 71: 2, 72: None, 73: None, 74: 9, 75: 5, 76: None, 77: 6, 78: 8, 79: 9
}

# Labels for individual images for the automated run to automate the active refinements stage
my_refinements = {
    # --- ITERATION 1 ---
    8631: 2, 9648: 9, 4910: 6, 9605: 8, 7617: 1, 3356: 8, 260: 1, 495: 3, 
    6895: 6, 8389: 6, 3530: 9, 7866: 1, 7888: 6, 9240: 8, 376: 2, 4940: 9, 
    5186: 4, 8933: 9, 2421: 8, 5308: 1, 7083: 1, 8910: 2, 5643: 8, 4929: 9, 
    6656: 2, 6338: 3, 4810: 8, 5164: 1, 7969: 2, 4368: 2, 4228: 2, 1466: 2, 
    4026: 2, 7875: 4, 6023: 6, 4747: 2, 7913: 1, 6890: 2, 4594: 8, 8432: 8,

    # --- ITERATION 2 ---
    3891: 6, 4105: 1, 2129: 1, 877: 8, 8349: 8, 8142: 8, 7530: 8, 5400: 6, 
    9263: 6, 2866: 6, 4640: 6, 1678: 8, 5033: 6, 883: 7, 5723: 8, 6814: 4, 
    3299: 2, 9697: 8, 4430: 2, 6715: 4, 9753: 4, 1269: 4, 5962: 2, 555: 2, 
    6218: 2, 3165: 2, 2696: 6, 7182: 4, 8597: 8, 2543: 2, 3239: 2, 7093: 2, 
    2791: 2, 8932: 9, 4087: 2, 8331: 6, 2927: 8, 6843: 2, 4635: 2, 5059: 4,

    # --- ITERATION 3 ---
    4863: 6, 114: 6, 942: 6, 7968: 8, 5384: 8, 5265: 6, 9983: 4, 9302: 4, 
    1199: 4, 8111: 8, 4455: 2, 8231: 2, 6025: 2, 1139: 2, 270: 6, 9231: 2, 
    9253: 2, 5808: 2, 1967: 4, 6069: 8, 6347: 4, 5478: 3, 8559: 2, 8021: 2, 
    1393: 8, 9919: 2, 6633: 3, 5007: 2, 3384: 4, 1340: 2, 5443: 9, 7870: 8, 
    1550: 2, 433: 4, 2472: 2, 7180: 7, 3260: 2, 4975: 2, 9609: 2, 7034: 6
}

# STAGE 1: LOADING & HOG EXTRACTION

# User Input to select between manual training or demonstration
print("Select Pipeline 1 Mode:")
print("[1] Automated Run (Uses saved cluster/refinement dictionaries)")
print("[2] Interactive Training (User must manually label clusters/refinements)")
mode_choice = input("Enter Choice (1 or 2): ")
auto_mode = True if mode_choice == '1' else False

def load_indian_digits(folder_path):
    """Reads .bmp images and flattens them into a numpy array."""
    print("\nLoading 10,000 images from folder...")
    images = []
    for i in range(1, 10001):
        img_path = os.path.join(folder_path, f"{i}.bmp")
        img = Image.open(img_path).convert('L')
        images.append(np.array(img).flatten())
    return np.array(images)

# Load the dataset
X_raw = load_indian_digits('F:/EECE 26/EECE - Year 4/_2nd_term/Neural Networks\Assignments\Assignment 01\Part3\Indian_Digits_Train') 
    
print("Extracting HOG Features...")
def get_hog(data):
    """Converts raw pixels to Histogram of Oriented Gradients (HOG) features."""
    features = []
    for img in data:
        f = hog(img.reshape(28,28), orientations=9, pixels_per_cell=(7,7), 
                cells_per_block=(2,2), block_norm='L2-Hys')
        features.append(f)
    return np.array(features)

# Extract and standardize features for K-Means and SVM
X_features = get_hog(X_raw)
X_scaled = StandardScaler().fit_transform(X_features)
print("HOG Extraction & Scaling complete.")

# STAGE 2: K-MEANS CLUSTERING & LABELLING

# group images by visual similarity to label them
K = 80
print(f"\nPerforming K-Means Clustering (K={K})...")
km = KMeans(n_clusters=K, n_init=10, random_state=42)
cluster_ids = km.fit_predict(X_scaled)

cluster_to_digit = {}
total_manual_time = 0

# Assign a digit class to each cluster based on visual samples
for cluster_idx in range(K):
    if auto_mode and cluster_idx in my_clusters:
        cluster_to_digit[cluster_idx] = my_clusters[cluster_idx]
        total_manual_time += 20  # Estimated manual time per cluster
    else:
        # Show a random sample of images from this cluster for user identification
        indices = np.where(cluster_ids == cluster_idx)[0]
        samples = np.random.choice(indices, min(8, len(indices)), replace=False)
        plt.figure(figsize=(10, 2))
        plt.suptitle(f"Cluster {cluster_idx} Samples", fontsize=14)
        for i, idx in enumerate(samples):
            plt.subplot(1, 8, i+1)
            plt.imshow(X_raw[idx].reshape(28,28), cmap='gray')
            plt.title(f"ID: {idx}")
            plt.axis('off')
        plt.show()
        label = input(f"Enter Digit for Cluster {cluster_idx} or 's' to skip: ")
        cluster_to_digit[cluster_idx] = int(label) if label.isdigit() else None
        total_manual_time += 20

# Propagate cluster labels to all individual images within those clusters
y_train_bootstrapped = np.full(10000, -1)
for cluster_idx, digit in cluster_to_digit.items():
    if digit is not None:
        indices = np.where(cluster_ids == cluster_idx)[0]
        y_train_bootstrapped[indices] = digit

# STAGE 3: INITIAL SVM TRAINING

# Train a classifier on the bootstrapped data from clustering.
print("\nTraining initial SVM on cluster-labeled data...")
train_mask = y_train_bootstrapped != -1
svm_model = SVC(kernel='rbf', C=100, probability=True, random_state=42)
svm_model.fit(X_scaled[train_mask], y_train_bootstrapped[train_mask])

# STAGE 4: ITERATIVE ACTIVE REFINEMENT

# Target ambiguous images to refine the SVM decision boundaries.

target_accuracy = 0.99
current_accuracy = 0.0
iteration = 1
manually_labeled_indices = [] 
weights = np.ones(10000) 

while current_accuracy < target_accuracy:
    print(f"\n--- ITERATION {iteration} ---")
    
    # Calculate confidence margins (higher margin = higher confidence)
    scores = svm_model.decision_function(X_scaled)
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]

    # Select the 40 images with the smallest margins (most ambiguous)
    sorted_indices = np.argsort(margins)
    uncertain_indices = [idx for idx in sorted_indices if idx not in manually_labeled_indices][:40]

    print(f"Refining {len(uncertain_indices)} uncertain images...")

    for idx in uncertain_indices:
        if auto_mode and idx in my_refinements:
            true_label = my_refinements[idx]
        else:
            # Manually label specific image for precision
            plt.figure(figsize=(2, 2))
            plt.imshow(X_raw[idx].reshape(28,28), cmap='gray')
            plt.title(f"Manual Label: Image {idx}")
            plt.axis('off')
            plt.show()
            true_label = int(input(f"Correct Digit for Image {idx} (0-9): "))
            my_refinements[idx] = true_label

        y_train_bootstrapped[idx] = true_label
        weights[idx] = 200 # high weight to human labels
        manually_labeled_indices.append(idx)
        total_manual_time += 10 # Estimated 10s per image for refinement

    print(f"Retraining SVM with refined labels...")
    # Retrain using sample weights to prioritize refined labels over cluster labels
    mask = y_train_bootstrapped != -1
    svm_model.fit(X_scaled[mask], y_train_bootstrapped[mask], sample_weight=weights[mask])

    # Evaluate performance against the ground truth oracle
    y_final_pred = svm_model.predict(X_scaled)
    current_accuracy, _, _ = check_accuracy(y_final_pred)
    
    print(f"Accuracy: {current_accuracy * 100:.2f}% | Manual Time: {total_manual_time / 60:.2f} min")

    iteration += 1

print("\n--- Pipeline 1 Complete ---")
print(f"Final Labelling Accuracy: {current_accuracy*100:.2f}%") 
print(f"Total Iterations: {iteration-1}") 
print(f"Total Manual Time: {total_manual_time/60:.2f} minutes")
