# Artificial Neural Networks: Optimizers Performance Analysis

## Overview

This project implements a complex Artificial Neural Network (ANN) using TensorFlow/Keras and evaluates the performance of different optimizers on the CIFAR-10 dataset. The analysis focuses on how different optimization algorithms affect training speed, convergence, and overall accuracy.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 different classes:
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

The dataset is split into 50,000 training images and 10,000 testing images.

## Model Architecture

A deep neural network with multiple dense layers was implemented:

```python
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])
```

## Optimizers Compared

The project evaluates five popular optimization algorithms:

1. **Stochastic Gradient Descent (SGD)**: Basic optimizer with constant learning rate
2. **SGD with Momentum**: Adds momentum to accelerate convergence
3. **Adam**: Adaptive learning rates with momentum and RMSprop properties
4. **RMSprop**: Adapts learning rates based on moving average of squared gradients
5. **Adagrad**: Adapts learning rates based on historical gradients

## Evaluation Metrics

Each optimizer was evaluated based on:
- Final validation accuracy
- Training time
- Convergence speed
- Loss reduction rate
- Generalization performance

## Results

### Accuracy Comparison
- SGD: 56.3%
- SGD with Momentum: 62.7%
- Adam: 68.2%
- RMSprop: 65.9%
- Adagrad: 61.4%

### Training Time Comparison
- SGD: 9.3 minutes
- SGD with Momentum: 9.5 minutes
- Adam: 10.2 minutes
- RMSprop: 10.0 minutes
- Adagrad: 9.7 minutes

## Key Findings

1. **Adam** achieved the highest accuracy and fastest convergence
2. **RMSprop** performed second best with good convergence properties
3. **SGD with Momentum** significantly outperformed basic SGD
4. **Adagrad** showed initial fast convergence but plateaued earlier
5. Basic **SGD** required more epochs to achieve comparable results

## Visualizations

The project includes:
- Loss and accuracy curves for each optimizer
- Comparative performance graphs
- Learning rate adaptation visualization
- Confusion matrices for each optimizer's predictions

## Conclusion

This study demonstrates the significant impact that optimizer choice has on neural network performance. For the CIFAR-10 image classification task with our ANN architecture, Adam provided the best balance of accuracy, convergence speed, and generalization. However, the optimal choice may vary depending on the specific application, dataset size, and model architecture.

## Future Work

- Test performance on different datasets and model architectures
- Experiment with learning rate schedules
- Compare with newer optimizers (Nadam, AdamW, etc.)
- Analyze optimizer behavior with different batch sizes
- Investigate the relationship between optimizers and regularization techniques 