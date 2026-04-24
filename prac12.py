import numpy as np

# ==================== CNN using TensorFlow ====================
print("="*60)
print("CNN Implementation using TensorFlow")
print("="*60)

import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
print("\nLoading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# Expand dimensions (add channel dimension)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Build CNN model
print("Building TensorFlow CNN model...")
model_tf = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    layers.Flatten(name='flatten'),
    layers.Dense(64, activation='relu', name='fc1'),
    layers.Dense(10, activation='softmax', name='output')
])

# Compile
model_tf.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

# Display model architecture
print("\nTensorFlow Model Architecture:")
model_tf.summary()

# Train
print("\nTraining TensorFlow CNN (3 epochs)...")
history_tf = model_tf.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)

# Evaluate
print("\nEvaluating TensorFlow CNN on test data...")
test_loss_tf, test_accuracy_tf = model_tf.evaluate(x_test, y_test, verbose=0)
print(f"TensorFlow CNN - Test Loss: {test_loss_tf:.4f}, Test Accuracy: {test_accuracy_tf:.4f}")


# ==================== CNN using PyTorch ====================
print("\n" + "="*60)
print("CNN Implementation using PyTorch")
print("="*60)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        # Conv layer 1 + Pool
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Conv layer 2 + Pool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Flatten
        x = x.view(x.size(0), -1)  # 64*7*7
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model, loss, and optimizer
print("\nBuilding PyTorch CNN model...")
model_pytorch = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pytorch.parameters(), lr=0.001)

# Display model architecture
print("\nPyTorch Model Architecture:")
print(model_pytorch)

# Prepare data
print("\nPreparing data...")
# Convert from (batch, height, width, channels) to (batch, channels, height, width)
x_train_pytorch = np.transpose(x_train, (0, 3, 1, 2))
x_test_pytorch = np.transpose(x_test, (0, 3, 1, 2))

x_train_tensor = torch.from_numpy(x_train_pytorch).float().to(device)
y_train_tensor = torch.from_numpy(y_train).long().to(device)
x_test_tensor = torch.from_numpy(x_test_pytorch).float().to(device)
y_test_tensor = torch.from_numpy(y_test).long().to(device)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Training loop
print("Training PyTorch CNN (2 epochs)...")
num_epochs = 2
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model_pytorch(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluate on test data
print("\nEvaluating PyTorch CNN on test data...")
model_pytorch.eval()
with torch.no_grad():
    test_outputs = model_pytorch(x_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy_pytorch = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"PyTorch CNN - Test Accuracy: {accuracy_pytorch:.4f}")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"TensorFlow CNN Test Accuracy: {test_accuracy_tf:.4f}")
print(f"PyTorch CNN Test Accuracy: {accuracy_pytorch:.4f}")
print("="*60)
