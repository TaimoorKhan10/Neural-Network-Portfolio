# CNN Image Classification on CIFAR-10


```markdown
# CNN Image Classification on CIFAR-10

This document implements a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset using TensorFlow/Keras. We'll explore how different architectures and optimization techniques affect model performance.

**Author:** Taimoor Khan  
**Date:** May 2023

## Overview

We will cover the following key aspects:
1. Loading and preprocessing the CIFAR-10 dataset
2. Building a CNN model with optimization techniques
3. Training the model with callbacks
4. Visualizing performance and results
5. Analyzing misclassified examples

## 1. Import Libraries

```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check if GPU is available
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU Devices: ", tf.config.list_physical_devices('GPU'))
```

## 2. Load and Preprocess CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

```python
# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Print shapes
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Test labels shape: {y_test.shape}")

# Define class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display some random samples
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    idx = np.random.randint(0, len(x_train))
    plt.imshow(x_train[idx])
    plt.xlabel(class_names[y_train[idx][0]])
plt.tight_layout()
plt.show()
```

### 2.1 Data Preprocessing

```python
# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

print(f"Original label for first sample: {y_train[0][0]} (class name: {class_names[y_train[0][0]]})")
print(f"One-hot encoded label: {y_train_onehot[0]}")

# Data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)
```

## 3. Build CNN Model

Our CNN architecture will include:
- Multiple convolutional blocks with batch normalization
- Dropout regularization
- Global average pooling
- Dense layers for classification

```python
def build_cnn_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Build the model
model = build_cnn_model()

# Display model summary
model.summary()
```

## 4. Train the Model

We'll train the model with the following callbacks:
1. Early stopping to prevent overfitting
2. Model checkpointing to save the best model
3. Learning rate reduction to improve convergence

```python
# Define callbacks
callbacks = [
    # Early stopping to prevent overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save the best model during training
    ModelCheckpoint(
        'best_cifar10_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    
    # Reduce learning rate when training stagnates
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train the model with data augmentation
batch_size = 64
epochs = 50  # We have early stopping, so we set a high number of epochs

history = model.fit(
    datagen.flow(x_train, y_train_onehot, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test_onehot),
    callbacks=callbacks
)
```

## 5. Evaluate Model Performance

```python
# Load the best model (if training was interrupted)
try:
    model = tf.keras.models.load_model('best_cifar10_model.h5')
    print("Loaded the best model from disk")
except:
    print("Using the last trained model")

# Evaluate model on test data
test_loss, test_acc = model.evaluate(x_test, y_test_onehot, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('cifar10_training_history.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 6. Confusion Matrix and Classification Report

```python
# Get predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_onehot, axis=1)

# Compute and plot confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification report
report = classification_report(y_true_classes, y_pred_classes, 
                               target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
display(report_df)

# Save the results to a CSV file
report_df.to_csv('classification_report.csv')
```

## 7. Visualize Misclassifications

Let's examine some examples that the model misclassified to understand its limitations.

```python
# Find misclassified examples
misclassified_indices = np.where(y_pred_classes != y_true_classes)[0]
print(f"Total misclassified examples: {len(misclassified_indices)} out of {len(x_test)} test images")

# Display random misclassified examples
num_to_display = min(25, len(misclassified_indices))
random_indices = np.random.choice(misclassified_indices, num_to_display, replace=False)

plt.figure(figsize=(12, 12))
for i, idx in enumerate(random_indices):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_test[idx])
    plt.title(f"True: {class_names[y_true_classes[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('misclassified_examples.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 8. Conclusion

In this analysis, we:
1. Built and trained a CNN for CIFAR-10 image classification
2. Applied optimization techniques like batch normalization and dropout
3. Used data augmentation to improve model generalization
4. Analyzed model performance with various visualization techniques
5. Explored model predictions and feature maps

The model achieved good performance on the CIFAR-10 dataset, demonstrating the effectiveness of CNNs for image classification tasks. Further improvements could include:
- Trying different architectures (ResNet, DenseNet, etc.)
- Tuning hyperparameters more extensively
- Implementing more advanced regularization techniques
- Using transfer learning with pre-trained models

## Future Work

In future extensions of this project, we could:
1. Implement other CNN architectures for comparison
2. Apply transfer learning with pre-trained models
3. Use techniques like Grad-CAM for better visualization of model decisions
4. Deploy the model for real-time classification 
```
