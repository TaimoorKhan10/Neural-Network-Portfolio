import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directory for results
if not os.path.exists('output'):
    os.makedirs('output')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms for the training and test sets
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def load_data(batch_size=128, num_workers=2):
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Data summary
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    print(f"Number of classes: {len(trainset.classes)}")
    print(f"Class names: {trainset.classes}")
    
    return trainloader, testloader, trainset.classes

# Define ResNet building blocks
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Create different ResNet variants
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

# Training function
def train(model, trainloader, optimizer, criterion, epoch, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': train_loss/(batch_idx+1), 
            'Acc': 100.*correct/total
        })

    return train_loss/len(trainloader), 100.*correct/total

# Testing function
def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss/len(testloader), 100.*correct/total

# Visualize model architecture complexity
def plot_model_complexity(models_dict):
    model_names = list(models_dict.keys())
    params = [sum(p.numel() for p in models_dict[name].parameters())/1_000_000 for name in model_names]  # Convert to millions
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, params)
    
    plt.title('Model Complexity Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Number of Parameters (millions)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add parameter count on top of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f"{params[i]:.2f}M", ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/model_complexity.png', dpi=300, bbox_inches='tight')
    plt.show()

# Visualize results
def plot_training_results(results):
    # Plot test accuracy vs epochs
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for model_name, result in results.items():
        plt.plot(range(1, len(result["test_acc"])+1), result["test_acc"], label=f"{model_name} (Final: {result['test_acc'][-1]:.2f}%)")
    plt.title('Test Accuracy vs Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot test loss vs epochs
    plt.subplot(2, 2, 2)
    for model_name, result in results.items():
        plt.plot(range(1, len(result["test_loss"])+1), result["test_loss"], label=model_name)
    plt.title('Test Loss vs Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot training accuracy vs epochs
    plt.subplot(2, 2, 3)
    for model_name, result in results.items():
        plt.plot(range(1, len(result["train_acc"])+1), result["train_acc"], label=model_name)
    plt.title('Training Accuracy vs Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot training loss vs epochs
    plt.subplot(2, 2, 4)
    for model_name, result in results.items():
        plt.plot(range(1, len(result["train_loss"])+1), result["train_loss"], label=model_name)
    plt.title('Training Loss vs Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot bar charts for performance metrics
def plot_performance_metrics(results):
    model_names = list(results.keys())
    
    # Extract performance metrics
    final_accuracies = [results[name]['test_acc'][-1] for name in model_names]
    training_times = [results[name]['training_time'] for name in model_names]
    inference_times = [results[name]['inference_time'] for name in model_names]
    
    # Plot performance metrics
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Final Test Accuracy
    axs[0].bar(model_names, final_accuracies, color='skyblue')
    axs[0].set_title('Final Test Accuracy', fontsize=14)
    axs[0].set_ylabel('Accuracy (%)', fontsize=12)
    # Add text on top of bars
    for i, acc in enumerate(final_accuracies):
        axs[0].text(i, acc + 0.2, f"{acc:.2f}%", ha='center', fontsize=10)
    axs[0].grid(axis='y', alpha=0.3)
    
    # Training Time
    axs[1].bar(model_names, training_times, color='orange')
    axs[1].set_title('Training Time', fontsize=14)
    axs[1].set_ylabel('Time (seconds)', fontsize=12)
    # Add text on top of bars
    for i, time_val in enumerate(training_times):
        axs[1].text(i, time_val + max(training_times)*0.02, f"{time_val:.1f}s", ha='center', fontsize=10)
    axs[1].grid(axis='y', alpha=0.3)
    
    # Inference Time
    axs[2].bar(model_names, inference_times, color='green')
    axs[2].set_title('Inference Time (per batch)', fontsize=14)
    axs[2].set_ylabel('Time (milliseconds)', fontsize=12)
    # Add text on top of bars
    for i, time_val in enumerate(inference_times):
        axs[2].text(i, time_val + max(inference_times)*0.05, f"{time_val:.2f}ms", ha='center', fontsize=10)
    axs[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot accuracy vs complexity scatter plot
def plot_accuracy_vs_complexity(results, models_dict):
    model_names = list(results.keys())
    final_accuracies = [results[name]['test_acc'][-1] for name in model_names]
    params = [sum(p.numel() for p in models_dict[name].parameters())/1_000_000 for name in model_names]  # Convert to millions
    
    plt.figure(figsize=(10, 6))
    plt.scatter(params, final_accuracies, s=200, alpha=0.7)
    
    # Add model names as labels
    for i, name in enumerate(model_names):
        plt.annotate(name, (params[i], final_accuracies[i]), fontsize=12)
    
    plt.title('Accuracy vs Model Complexity', fontsize=16)
    plt.xlabel('Number of Parameters (millions)', fontsize=14)
    plt.ylabel('Test Accuracy (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/accuracy_vs_complexity.png', dpi=300, bbox_inches='tight')
    plt.show()

# Create summary table of results
def create_summary_table(results, models_dict):
    data = {
        'Model': [],
        'Test Accuracy (%)': [],
        'Parameters (M)': [],
        'Training Time (s)': [],
        'Inference Time (ms)': [],
        'Convergence Rate': []
    }
    
    for name in results.keys():
        # Calculate convergence rate (epochs to reach 90% of max accuracy)
        max_acc = results[name]['test_acc'][-1]
        convergence_threshold = 0.9 * max_acc
        epochs_to_converge = next((i+1 for i, acc in enumerate(results[name]['test_acc']) if acc >= convergence_threshold), len(results[name]['test_acc']))
        
        data['Model'].append(name)
        data['Test Accuracy (%)'].append(f"{results[name]['test_acc'][-1]:.2f}")
        data['Parameters (M)'].append(f"{sum(p.numel() for p in models_dict[name].parameters())/1_000_000:.2f}")
        data['Training Time (s)'].append(f"{results[name]['training_time']:.1f}")
        data['Inference Time (ms)'].append(f"{results[name]['inference_time']:.2f}")
        data['Convergence Rate'].append(f"{epochs_to_converge} epochs")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('output/resnet_comparison_results.csv', index=False)
    
    return df

# Visualize confusion matrices
def plot_confusion_matrices(results, class_names):
    plt.figure(figsize=(18, 6))
    
    for i, (model_name, result) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('output/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Define hyperparameters
    batch_size = 128
    num_epochs = 50
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    
    # Load data
    trainloader, testloader, class_names = load_data(batch_size)
    
    # Define models to compare
    models = {
        "ResNet18": ResNet18(),
        "ResNet34": ResNet34(),
        "ResNet50": ResNet50()
    }
    
    # Move models to device
    for name, model in models.items():
        models[name] = model.to(device)
        print(f"{name} parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Plot model complexity
    plot_model_complexity(models)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Store results
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*50}\nTraining {model_name}...\n{'='*50}")
        
        # Initialize optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # Initialize result tracking
        results[model_name] = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "confusion_matrix": None,
            "training_time": 0,
            "inference_time": 0
        }
        
        # Start training timer
        start_time = time.time()
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = train(model, trainloader, optimizer, criterion, epoch, device)
            results[model_name]["train_loss"].append(train_loss)
            results[model_name]["train_acc"].append(train_acc)
            
            # Test
            test_loss, test_acc = test(model, testloader, criterion, device)
            results[model_name]["test_loss"].append(test_loss)
            results[model_name]["test_acc"].append(test_acc)
            
            # Update learning rate
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Record training time
        results[model_name]["training_time"] = time.time() - start_time
        
        # Measure inference time
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for inputs, _ in testloader:
                inputs = inputs.to(device)
                _ = model(inputs)
                break  # Just measure one batch
        results[model_name]["inference_time"] = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Generate confusion matrix
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        results[model_name]["confusion_matrix"] = np.zeros((10, 10), dtype=int)
        for i in range(len(y_true)):
            results[model_name]["confusion_matrix"][y_true[i], y_pred[i]] += 1
        
        # Save model
        torch.save(model.state_dict(), f'output/{model_name}_cifar10.pth')
        print(f"{model_name} saved, training time: {results[model_name]['training_time']:.2f}s")
    
    # Visualize results
    plot_training_results(results)
    plot_performance_metrics(results)
    plot_accuracy_vs_complexity(results, models)
    plot_confusion_matrices(results, class_names)
    
    # Create summary table
    summary_df = create_summary_table(results, models)
    print("\nResults Summary:")
    print(summary_df)
    
    print("\nResNet architecture comparison completed successfully!")

if __name__ == "__main__":
    main()
