import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the paths to the MNIST dataset files
data_path = './data'
train_images_file = 'train-images-idx3-ubyte'
train_labels_file = 'train-labels-idx1-ubyte'
test_images_file = 't10k-images-idx3-ubyte'
test_labels_file = 't10k-labels-idx1-ubyte'


# Function to load MNIST dataset from local files
def load_mnist(data_path, images_file, labels_file):
    images_path = os.path.join(data_path, images_file)
    labels_path = os.path.join(data_path, labels_file)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

# Function to visualize MNIST samples
def visualize_samples(images, labels, num_samples=10):
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Function to visualize model predictions
def visualize_predictions(model, images, labels, num_samples=10):
    model.eval()
    with torch.no_grad():
        outputs = model(images[:num_samples])
        _, predicted = torch.max(outputs.data, 1)

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i in range(num_samples):
        ax = axes[i]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Predicted: {predicted[i]}\nActual: {labels[i]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Load the MNIST dataset from local files
train_images, train_labels = load_mnist(data_path, train_images_file, train_labels_file)
test_images, test_labels = load_mnist(data_path, test_images_file, test_labels_file)

# Create PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_images).float(),
                                               torch.from_numpy(train_labels).long())
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_images).float(),
                                              torch.from_numpy(test_labels).long())


# Define the MLP model architecture
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # Input layer: 784 (28x28) inputs, 512 outputs
        self.fc2 = nn.Linear(512, 256)  # Hidden layer: 512 inputs, 256 outputs
        self.fc3 = nn.Linear(256, 10)  # Output layer: 256 inputs, 10 outputs (one for each digit class)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input images
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first linear layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second linear layer
        x = self.fc3(x)  # No activation for the output layer
        return x


# Create data loaders for training and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the MLP model
model = MNISTClassifier()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Visualize some training samples
visualize_samples(train_images[:10], train_labels[:10])

# Training loop
epochs = 10
for epoch in range(epochs):
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}')

# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

test_images_tensor = torch.from_numpy(test_images[:10]).float()
test_labels_tensor = torch.from_numpy(test_labels[:10]).long()
visualize_predictions(model, test_images_tensor, test_labels_tensor)

