
# CNN for CIFAR-10 Image Classification
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

<img src="https://miro.medium.com/v2/resize:fit:1400/1*vkQ0hXDaQv57sALXAJquxA.jpeg" width="600" alt="CNN Architecture">

## Overview

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is built with TensorFlow/Keras and includes advanced techniques like Batch Normalization and Dropout to improve performance.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

Each class contains 6,000 images, with 50,000 for training and 10,000 for testing.

<img src="https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png" width="500" alt="CIFAR-10 Samples">

## Model Architecture

The CNN architecture includes:
- Multiple Conv2D layers with increasing filter sizes (64, 128, 256)
- Batch Normalization after each convolutional layer
- MaxPooling2D layers to reduce spatial dimensions
- Dropout layers (0.5) to prevent overfitting
- Dense layers with ReLU activation
- Softmax output layer for 10-class classification

```python
model = Sequential([
    Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

<img src="https://miro.medium.com/v2/resize:fit:1400/1*ZCjPUFrB6eHPRi4eyP6aaA.gif" width="500" alt="CNN Operation">

## Training

The model was trained with:
- Adam optimizer (learning rate = 0.001)
- Categorical cross-entropy loss
- Accuracy metric
- Early stopping to prevent overfitting

## Results

The model achieved over 70% accuracy on the CIFAR-10 test dataset, demonstrating the effectiveness of the CNN architecture for image classification tasks.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*3fA77_mLNiJTSgZFhYnU0Q.png" width="500" alt="Training Results">

## Visualizations

The project includes visualizations of:
- Training vs. validation accuracy
- Confusion matrix for model evaluation
- Examples of correctly and incorrectly classified images

## Key Findings

- Batch Normalization significantly improved training stability
- Dropout layers (0.5) helped prevent overfitting
- Increasing the number of filters in deeper layers captured more complex features
- The model performed best on classes with distinctive features (ships, automobiles) and struggled more with visually similar classes (cats, dogs)

## Future Improvements

- Implement data augmentation to improve generalization
- Experiment with different CNN architectures (ResNet, MobileNet)
- Use transfer learning with pre-trained models
- Fine-tune hyperparameters (learning rate, batch size)
```

 
