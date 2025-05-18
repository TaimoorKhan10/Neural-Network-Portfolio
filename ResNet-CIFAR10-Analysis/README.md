# ResNet Architecture Analysis on CIFAR-10
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)


## Overview

This project compares the performance of different ResNet architectures (ResNet-18, ResNet-34, and ResNet-50) on the CIFAR-10 dataset. The goal is to analyze how network depth affects accuracy, training time, and overall performance on image classification tasks.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 categories:
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

<img src="https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png" width="500" alt="CIFAR-10 Samples">

## ResNet Architectures

This project implements and compares three ResNet variants:

1. **ResNet-18**: 18 layers with 11.7 million parameters
2. **ResNet-34**: 34 layers with 21.8 million parameters
3. **ResNet-50**: 50 layers with 25.6 million parameters

All models use residual connections to address the vanishing gradient problem in deep networks.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*6hF97Upuqg_LdsqWY6n_wg.png" width="500" alt="ResNet Skip Connection">

## Implementation

The models were implemented using PyTorch and trained with the following specifications:

```python
def get_resnet_model(version):
    if version == 18:
        model = models.resnet18(pretrained=False)
    elif version == 34:
        model = models.resnet34(pretrained=False)
    elif version == 50:
        model = models.resnet50(pretrained=False)
    else:
        raise ValueError("Unsupported ResNet version")

    # Modify final fully connected layer for 10 CIFAR-10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)
```

## Training Details

Each model was trained with the following parameters:
- Optimizer: Adam (learning rate = 0.001)
- Loss function: Cross Entropy Loss
- Batch size: 64
- Training epochs: 5
- GPU acceleration using CUDA


## Results

### Accuracy Comparison
- ResNet-18: 85.7%
- ResNet-34: 88.3%
- ResNet-50: 91.2%

### Training Time Comparison
- ResNet-18: 15.5 minutes
- ResNet-34: 27.2 minutes
- ResNet-50: 42.8 minutes


### Key Findings

1. **Accuracy vs. Depth**: Deeper networks generally achieved higher accuracy, with ResNet-50 performing the best.
2. **Training Time**: Deeper networks required significantly more training time.
3. **Efficiency-Performance Tradeoff**: ResNet-34 offered a good balance between accuracy and training efficiency.
4. **Convergence Rate**: Deeper networks showed slower initial convergence but ultimately reached higher accuracy.

## Visualizations

The project includes:
- Learning curves for each model
- Comparative accuracy graphs
- Confusion matrices
- Inference time analysis

## Conclusion

ResNet-50 achieved the highest accuracy but required substantially more computational resources. For applications with limited resources, ResNet-34 provides a good compromise between accuracy and efficiency. The results demonstrate the benefits of deeper architectures in capturing complex features for image classification tasks.

## Future Work

- Implement additional ResNet variants (ResNet-101, ResNet-152)
- Apply transfer learning with pre-trained weights
- Experiment with different optimizers and learning rates
- Implement data augmentation to improve generalization
- Explore model pruning to reduce parameters while maintaining accuracy 
