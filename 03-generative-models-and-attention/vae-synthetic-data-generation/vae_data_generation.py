import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. DATA PREPARATION ---
full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

indices = []
for digit in range(10):
    digit_indices = (full_train.targets == digit).nonzero(as_tuple=True)[0]
    indices.extend(digit_indices[:350].tolist())

real_350_data = Subset(full_train, indices)

# Augmentation Pipeline (ToTensor removed because Subset already provides Tensors)
augment_pipeline = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)) # to prevent values from exceeding the range [0, 1]
])

augmented_data_list = []
for img, label in real_350_data:
    augmented_data_list.append((img, label)) # Include original
    for _ in range(14): # Create 14 more variations per image
        # bec pytorch expects a 4D tensor so we add a dimension then remove it
        aug_img = augment_pipeline(img.unsqueeze(0)).squeeze(0) 
        augmented_data_list.append((aug_img, label))

train_loader_vae = DataLoader(augmented_data_list, batch_size=64, shuffle=True)
# Loader for training the Judge (Real data only)
train_loader_judge = DataLoader(real_350_data, batch_size=64, shuffle=True)

# --- 2. CVAE DEFINITION ---
class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, num_classes):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(feature_size + num_classes, 400) # conditional = concat the label to the input (784+10)
        self.fc2_mu = nn.Linear(400, latent_size)
        self.fc2_logvar = nn.Linear(400, latent_size)
        self.fc3 = nn.Linear(latent_size + num_classes, 400) # conditional = concat the label to the input (20+10)
        self.fc4 = nn.Linear(400, feature_size)

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = torch.relu(self.fc1(inputs))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # sigma=exp(0.5*ln(sigma^2))
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = torch.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3)) # to make every pixel between [0, 1]

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 784), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KLD = -0.5*sum(1+ln(sigma^2)-mean^2-sigma^2)
    return BCE + KLD

# --- 3. TRAIN CVAE (Multi-run Generation) ---
all_synthetic_images = []
all_synthetic_labels = []

# 5x runs of 1000 examples per digit
for run in range(5):
    print(f"Starting VAE Run {run+1}/5...")
    
    # 1. Initialize new weights for this run
    run_model = CVAE(feature_size=784, latent_size=20, num_classes=10).to(device)
    optimizer = optim.Adam(run_model.parameters(), lr=1e-3)
    
    # 2. Train for this specific run
    run_model.train()
    for epoch in range(30): 
        for img, label in train_loader_vae:
            img = img.to(device)
            label_oh = F.one_hot(label, 10).float().to(device)
            recon, mu, logvar = run_model(img, label_oh)
            loss = vae_loss(recon, img, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 3. Generate 1000 samples per digit for this weight variant
    run_model.eval()
    with torch.no_grad():
        for digit in range(10):
            z = torch.randn(1000, 20).to(device)
            labels = torch.full((1000,), digit, dtype=torch.long).to(device)
            c = F.one_hot(labels, 10).float().to(device)
            samples = run_model.decode(z, c)
            
            all_synthetic_images.append(samples.cpu())
            all_synthetic_labels.append(labels.cpu())

# Combine everything into Set A (50,000 total images)
set_a_images = torch.cat(all_synthetic_images) 
set_a_labels = torch.cat(all_synthetic_labels)

# --- 4. TRAIN JUDGE (LENET-5) ---
# We must train the judge on ONLY the 350 real samples first
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

judge_lenet = LeNet5().to(device)
judge_optimizer = optim.Adam(judge_lenet.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

judge_lenet.train()
for epoch in range(10):
    for img, label in train_loader_judge:
        img, label = img.to(device), label.to(device)
        output = judge_lenet(img)
        loss = criterion(output, label)
        judge_optimizer.zero_grad()
        loss.backward()
        judge_optimizer.step()

# --- 5. FILTERING ---
judge_lenet.eval()

# Reshape Set A to 28x28 for the Judge
set_a_images_reshaped = set_a_images.view(-1, 1, 28, 28)

confidences_list = []
with torch.no_grad():
    # Process in batches to avoid memory issues with 50,000 images
    for i in range(0, len(set_a_images_reshaped), 500):
        batch = set_a_images_reshaped[i:i+500].to(device)
        output = judge_lenet(batch)
        probs = torch.softmax(output, dim=1) # probability for each class
        conf, _ = torch.max(probs, dim=1) # pick the highest probability
        confidences_list.append(conf.cpu())

confidences = torch.cat(confidences_list)

# Set B indices (High Confidence)
set_b_mask = (confidences >= 0.9)
set_b_images = set_a_images[set_b_mask]
set_b_labels = set_a_labels[set_b_mask]

# Set C indices (Mid Confidence/Diverse)
set_c_mask = (confidences >= 0.6) & (confidences <= 0.9)
set_c_images = set_a_images[set_c_mask]
set_c_labels = set_a_labels[set_c_mask]

print(f"Set A (All): {len(set_a_images)} images")
print(f"Set B (High Conf): {len(set_b_images)} images")
print(f"Set C (Mid Conf): {len(set_c_images)} images")

# --- 6. FINAL EVALUATION: COMPARISON ---
def evaluate_lenet(train_images, train_labels, test_loader):
    """Trains a fresh LeNet from scratch and returns test accuracy."""
    model = LeNet5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset and loader
    dataset = torch.utils.data.TensorDataset(train_images.view(-1, 1, 28, 28), train_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Train for 10 epochs
    model.train()
    for epoch in range(10):
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            # model(img) -> model makes a guess - then compares to actual labels
            loss = criterion(model(img), lbl) 
            loss.backward() # math to figure out how to fix the weights
            optimizer.step() # actually updating the weights
            
    # Test
    model.eval()
    correct = 0
    with torch.no_grad(): # turn off learning mode for testing
        for img, lbl in test_loader:
            img, lbl = img.to(device), lbl.to(device)
            # picks the index of the highest score (the label)
            preds = model(img).argmax(dim=1) 
            correct += (preds == lbl).sum().item()
    
    return (correct / len(test_loader.dataset)) * 100, model

# Load standard MNIST test set
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transforms.ToTensor()), batch_size=1000)

# Extract real 350 images as Tensors for the combination
real_imgs = torch.stack([real_350_data[i][0] for i in range(len(real_350_data))])
real_lbls = torch.tensor([real_350_data[i][1] for i in range(len(real_350_data))])

# 1. Baseline: Real 350 only
acc_baseline, model_baseline = evaluate_lenet(real_imgs, real_lbls, test_loader)

# 2. Experiment A: Real 350 + Set A (All Generated)
acc_a, _ = evaluate_lenet(torch.cat([real_imgs, set_a_images.view(-1, 1, 28, 28)]), 
                       torch.cat([real_lbls, set_a_labels]), test_loader)

# 3. Experiment B: Real 350 + Set B (High Conf)
# WE SAVE THIS MODEL AS 'best_model'
acc_b, model_b = evaluate_lenet(torch.cat([real_imgs, set_b_images.view(-1, 1, 28, 28)]), 
                       torch.cat([real_lbls, set_b_labels]), test_loader)

# 4. Experiment C: Real 350 + Set C (Mid/Diverse)
acc_c, _ = evaluate_lenet(torch.cat([real_imgs, set_c_images.view(-1, 1, 28, 28)]), 
                       torch.cat([real_lbls, set_c_labels]), test_loader)

print(f"\nFinal Results:")
print(f"350 Real Baseline: {acc_baseline:.2f}%")
print(f"Set A (All VAE): {acc_a:.2f}%")
print(f"Set B (CL > 0.9): {acc_b:.2f}%")
print(f"Set C (0.6 < CL < 0.9): {acc_c:.2f}%")

# Get 1000 real indices per digit
indices_1000 = []
for digit in range(10):
    digit_indices = (full_train.targets == digit).nonzero(as_tuple=True)[0]
    indices_1000.extend(digit_indices[:1000].tolist())

# Create the dataset
real_1000_data = Subset(full_train, indices_1000)

# Extract as Tensors
imgs_1000 = torch.stack([real_1000_data[i][0] for i in range(len(real_1000_data))])
lbls_1000 = torch.tensor([real_1000_data[i][1] for i in range(len(real_1000_data))])

# Run the evaluation
acc_1000_baseline, _ = evaluate_lenet(imgs_1000, lbls_1000, test_loader)
print(f"1000 Real Baseline: {acc_1000_baseline:.2f}%")

# --- 7. VISUALIZATION ---
def plot_samples(images, labels, title, num_examples=3):
    fig, axes = plt.subplots(num_examples, 10, figsize=(15, num_examples * 1.5))
    fig.suptitle(title, fontsize=16)
    
    for digit in range(10):
        # Filter images of the current digit
        digit_mask = (labels == digit)
        digit_images = images[digit_mask]
        
        # Take the first few examples available
        for i in range(min(num_examples, len(digit_images))):
            ax = axes[i, digit]
            img = digit_images[i].view(28, 28).numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if i == 0:
                ax.set_title(str(digit))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Visualize Set B (High Confidence)
if len(set_b_images) > 0:
    plot_samples(set_b_images, set_b_labels, "Set B: High Confidence Samples (CL > 0.9)")

# Visualize Set C (Mid Confidence/Diverse)
if len(set_c_images) > 0:
    plot_samples(set_c_images, set_c_labels, "Set C: Mid Confidence Samples (0.6 < CL < 0.9)")


# --- 8. MORE VISUALIZATIONS ---

def plot_final_results(results_dict):
    """Generates a bar chart of the final accuracies."""
    plt.figure(figsize=(10, 6))
    names = list(results_dict.keys())
    values = list(results_dict.values())
    
    bars = plt.bar(names, values, color=['skyblue', 'salmon', 'lightgreen', 'orange', 'gold'])
    plt.ylim(min(values) - 2, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('Performance Comparison: VAE-Augmented vs. Baselines')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center')
    
    plt.show()

def plot_confusion_matrix(model1, model2, test_loader, device, title1="350 Real Baseline", title2="Set B (High Confidence)"):
    """Evaluates two models and plots their confusion matrices side-by-side."""
    model1.eval()
    model2.eval()
    
    all_preds1 = []
    all_preds2 = []
    all_labels = []
    
    # 1. Gather predictions for both models in a single pass
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            
            # Get predictions for both
            preds1 = model1(imgs).argmax(dim=1).cpu().numpy()
            preds2 = model2(imgs).argmax(dim=1).cpu().numpy()
            
            all_preds1.extend(preds1)
            all_preds2.extend(preds2)
            all_labels.extend(lbls.numpy())
            
    # 2. Calculate both matrices
    cm1 = confusion_matrix(all_labels, all_preds1)
    cm2 = confusion_matrix(all_labels, all_preds2)
    
    # 3. Create the side-by-side plot layout (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    
    # Plot Model 1 (Left)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(title1, fontsize=14)
    axes[0].set_xlabel('Predicted Label', fontsize=10)
    axes[0].set_ylabel('True Label', fontsize=10)
    
    # Plot Model 2 (Right)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(title2, fontsize=14)
    axes[1].set_xlabel('Predicted Label', fontsize=10)
    axes[1].set_ylabel('True Label', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Execution
results = {
    "350 Real": acc_baseline,
    "Set A (All)": acc_a,
    "Set B (High)": acc_b,
    "Set C (Mid)": acc_c,
    "1000 Real": acc_1000_baseline
}

plot_final_results(results)

plot_confusion_matrix(model_baseline, model_b, test_loader, device)


# normalization?
# sigmoid or softmax?
# MSE or BCE
# confusion matrix/graphs?
