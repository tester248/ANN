import numpy as np

print("="*70)
print("MNIST Handwritten Character Detection")
print("Comparison: TensorFlow, Keras, and PyTorch")
print("="*70)

# ==================== TensorFlow Implementation ====================
print("\n" + "="*70)
print("1. TensorFlow Implementation")
print("="*70)

import tensorflow as tf
from tensorflow.keras import layers, models

print("\nLoading MNIST dataset...")
mnist = tf.keras.datasets.mnist
(x_train_tf, y_train_tf), (x_test_tf, y_test_tf) = mnist.load_data()

print(f"Dataset shape - Training: {x_train_tf.shape}, Test: {x_test_tf.shape}")

# Normalize
x_train_tf = x_train_tf / 255.0
x_test_tf = x_test_tf / 255.0

# Build TensorFlow model
print("\nBuilding TensorFlow model...")
model_tf = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu', name='hidden'),
    layers.Dense(10, activation='softmax', name='output')
])

model_tf.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

print("\nTensorFlow Model Summary:")
model_tf.summary()

print("\nTraining TensorFlow model (3 epochs)...")
history_tf = model_tf.fit(x_train_tf, y_train_tf, epochs=3, batch_size=128, verbose=1)

print("\nEvaluating TensorFlow model...")
loss_tf, accuracy_tf = model_tf.evaluate(x_test_tf, y_test_tf, verbose=0)
print(f"TensorFlow - Test Loss: {loss_tf:.4f}, Test Accuracy: {accuracy_tf:.4f}")


# ==================== Keras Implementation ====================
print("\n" + "="*70)
print("2. Keras (Integrated with TensorFlow) Implementation")
print("="*70)

from tensorflow import keras
from tensorflow.keras import layers

print("\nLoading MNIST dataset...")
(x_train_keras, y_train_keras), (x_test_keras, y_test_keras) = keras.datasets.mnist.load_data()

# Normalize
x_train_keras = x_train_keras / 255.0
x_test_keras = x_test_keras / 255.0

# Build Keras model
print("Building Keras model...")
model_keras = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu', name='hidden'),
    layers.Dense(10, activation='softmax', name='output')
])

model_keras.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

print("\nKeras Model Summary:")
model_keras.summary()

print("\nTraining Keras model (3 epochs)...")
history_keras = model_keras.fit(x_train_keras, y_train_keras, epochs=3, batch_size=128, verbose=1)

print("\nEvaluating Keras model...")
loss_keras, accuracy_keras = model_keras.evaluate(x_test_keras, y_test_keras, verbose=0)
print(f"Keras - Test Loss: {loss_keras:.4f}, Test Accuracy: {accuracy_keras:.4f}")


# ==================== PyTorch Implementation ====================
print("\n" + "="*70)
print("3. PyTorch Implementation")
print("="*70)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

print("\nLoading MNIST dataset...")
mnist_torch = keras.datasets.mnist
(x_train_pt, y_train_pt), (x_test_pt, y_test_pt) = mnist_torch.load_data()

# Normalize
x_train_pt = x_train_pt / 255.0
x_test_pt = x_test_pt / 255.0

# Define PyTorch model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# Initialize model, loss, and optimizer
print("\nBuilding PyTorch model...")
model_pt = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pt.parameters(), lr=0.001)

print("\nPyTorch Model Architecture:")
print(model_pt)

# Prepare data loaders
print("\nPreparing PyTorch data loaders...")
x_train_pt_tensor = torch.from_numpy(x_train_pt).float().to(device)
y_train_pt_tensor = torch.from_numpy(y_train_pt).long().to(device)
x_test_pt_tensor = torch.from_numpy(x_test_pt).float().to(device)
y_test_pt_tensor = torch.from_numpy(y_test_pt).long().to(device)

train_dataset = TensorDataset(x_train_pt_tensor, y_train_pt_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = TensorDataset(x_test_pt_tensor, y_test_pt_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Training loop
print("\nTraining PyTorch model (3 epochs)...")
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model_pt(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Evaluate on test data
print("\nEvaluating PyTorch model...")
model_pt.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model_pt(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy_pt = 100 * correct / total
print(f"PyTorch - Test Accuracy: {accuracy_pt:.2f}%")


# ==================== Comparison and Summary ====================
print("\n" + "="*70)
print("Performance Comparison Summary")
print("="*70)
print(f"\nTensorFlow  - Test Accuracy: {accuracy_tf*100:.2f}%")
print(f"Keras       - Test Accuracy: {accuracy_keras*100:.2f}%")
print(f"PyTorch     - Test Accuracy: {accuracy_pt:.2f}%")
print("\n" + "="*70)
print("All three frameworks successfully trained on MNIST dataset!")
print("Framework Characteristics:")
print("  • TensorFlow: High-level API, easy to use, production-ready")
print("  • Keras: User-friendly, integrated with TensorFlow")
print("  • PyTorch: Flexible, dynamic computation graphs, research-friendly")
print("="*70)
