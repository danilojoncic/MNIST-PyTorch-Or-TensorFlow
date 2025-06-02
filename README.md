# PyTorch vs TensorFlow: FashionMNIST CNN Implementation Comparison

## Overview

Both implementations create identical CNN architectures with three convolutional blocks followed by fully connected layers to classify FashionMNIST images into 10 categories.

### Model Structure
Both models follow the same architecture:
- **Input**: 28×28 grayscale images
- **Conv Block 1**: 32 filters, 3×3 kernel, BatchNorm, ReLU, MaxPool
- **Conv Block 2**: 64 filters, 3×3 kernel, BatchNorm, ReLU, MaxPool  
- **Conv Block 3**: 128 filters, 3×3 kernel, BatchNorm, ReLU, MaxPool
- **FC Layers**: 256 → 128 → 10 neurons with Dropout (0.5)
- **Output**: 10 classes with softmax activation

## Code Structure Comparison

### 1. Library Imports

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
```

### 3. Data Loading & Preprocessing

**PyTorch:**
```python
# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset loading
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**TensorFlow:**
```python
# Direct dataset loading
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Manual preprocessing
x_train = x_train.astype('float32') / 255.0
x_train = (x_train - 0.5) / 0.5
x_train = np.expand_dims(x_train, -1)
y_train_categorical = keras.utils.to_categorical(y_train, 10)
```

### 4. Model Definition

**PyTorch (Class-based):**
```python
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        # ... 
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # ... 
        return x

model = FashionCNN().to(device)
```

**TensorFlow (Functional):**
```python
def create_fashion_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        # ... 
    ])
    return model

model = create_fashion_cnn()
```

### 5. Model Compilation/Setup

**PyTorch:**
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**TensorFlow:**
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 6. Training Loop

**PyTorch (Manual loop):**
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
```

**TensorFlow (Built-in):**
```python
history = model.fit(
    x_train, y_train_categorical,
    batch_size=64,
    epochs=10,
    callbacks=[TrainingCallback()],
    verbose=1
)
```

### 7. Model Evaluation

**PyTorch:**
```python
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        # Calculate accuracy
```

**TensorFlow:**
```python
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical)
```

### 8. Model Saving

**PyTorch:**
```python
torch.save(model.state_dict(), 'fashion_cnn_model.pth')
```

**TensorFlow:**
```python
model.save('fashion_cnn_model_tensorflow.h5')
```

## Key Differences

### Implementation Details

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| **Model Definition** | Class inheritance (`nn.Module`) | Sequential/Functional API |
| **Training Loop** | Manual implementation required | Built-in `model.fit()` |
| **Data Loading** | `DataLoader` with custom datasets | Built-in datasets + manual preprocessing |
| **Device Management** | Explicit `.to(device)` calls | Automatic GPU utilization |
| **Loss Functions** | Raw logits with `CrossEntropyLoss` | One-hot encoded with `categorical_crossentropy` |
| **Gradients** | Manual `zero_grad()` and `backward()` | Automatic in `model.fit()` |

### Advantages & Disadvantages

#### PyTorch Advantages
- **Flexibility**: Full control over training process
- **Debugging**: Native Python debugging support
- **Research-friendly**: Easy to implement custom architectures
- **Dynamic graphs**: Runtime graph modification

#### PyTorch Disadvantages  
- **Verbosity**: More boilerplate code required
- **Manual management**: Must handle device placement, gradient updates
- **Production deployment**: Requires additional tools

#### TensorFlow Advantages
- **Simplicity**: High-level APIs reduce code complexity
- **Production-ready**: Better deployment ecosystem
- **Built-in features**: Automatic differentiation, distributed training
- **Visualization**: Integrated TensorBoard support

#### TensorFlow Disadvantages
- **Less flexibility**: Harder to implement custom training loops
- **Debugging complexity**: Graph-based debugging can be challenging
- **API changes**: Frequent API updates between versions

## Performance Considerations

Both implementations should achieve similar performance since they use identical architectures. However:

- **PyTorch**: More fine-grained control over memory usage and computation
- **TensorFlow**: Better optimization for production deployment and serving
- **GPU Utilization**: Both frameworks efficiently utilize CUDA when available

## Conclusion

Both PyTorch and TensorFlow are excellent choices for deep learning projects. PyTorch offers more flexibility and control, making it ideal for research and custom implementations. TensorFlow provides higher-level abstractions and better production tools, making it suitable for deployment-focused projects.

The choice between them often depends on:
- Team expertise and preferences
- Project requirements (research vs. production)
- Ecosystem and community support
- Integration with existing tools and workflows

For learning purposes, both frameworks provide valuable insights into deep learning concepts, with PyTorch offering more visibility into the underlying mechanics and TensorFlow providing a more streamlined development experience.