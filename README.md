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

### 1. Data Loading & Preprocessing

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

### 2. Model Definition

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

### 3. Model Compilation/Setup

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

### 4. Training Loop

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

### 5. Model Evaluation

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

Both implementations achieved similar results  on the test set without any hyperparameter tuning or special data set tricks/manipulations
- **PyTorch**: 91.99%
- **TensorFlow**: 91.63%

Bear in mind the following training considerations
- **Number of Epochs** : 10
- **Batch Size** : 64


## Conclusion

Both PyTorch and TensorFlow are great. PyTorch offers more flexibility and control while TensorFlow provides higher-level abstractions. Based on previous experience using TensorFlow i would stick with using it instead of trying out PyTorch.