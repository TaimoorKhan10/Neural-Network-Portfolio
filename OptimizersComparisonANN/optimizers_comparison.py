
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load and preprocess CIFAR-10 dataset
def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to range [0,1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Convert labels to categorical format
    y_train, y_test = y_train.flatten(), y_test.flatten()
    
    # Flatten input data for fully connected network
    x_train_flat = x_train.reshape(-1, 32*32*3)
    x_test_flat = x_test.reshape(-1, 32*32*3)
    
    print(f"Training Data Shape: {x_train_flat.shape}")
    print(f"Testing Data Shape: {x_test_flat.shape}")
    
    return x_train_flat, y_train, x_test_flat, y_test

# Define complex neural network model
def create_complex_model():
    model = keras.Sequential([
        keras.layers.Dense(2048, activation='relu', input_shape=(32*32*3,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(), 
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])
    
    return model

# Dictionary of optimizers to compare
def get_optimizers(learning_rate=0.01):
    return {
        "SGD": SGD(learning_rate=learning_rate),
        "Momentum_SGD": SGD(learning_rate=learning_rate, momentum=0.9),
        "Adagrad": Adagrad(learning_rate=learning_rate),
        "RMSprop": RMSprop(learning_rate=learning_rate),
        "Adam": Adam(learning_rate=learning_rate)
    }

# Train the model with a specific optimizer
def train_and_evaluate(optimizer_name, optimizer, x_train, y_train, x_test, y_test, epochs=15, batch_size=64):
    print(f"\n{'='*50}")
    print(f"Training with {optimizer_name} optimizer...")
    print(f"{'='*50}")
    
    # Create new model instance
    model = create_complex_model()
    
    # Compile model with specified optimizer
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Measure training time
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\n{optimizer_name} Results:")
    print(f"- Test Accuracy: {test_accuracy:.4f}")
    print(f"- Training Time: {training_time:.2f} seconds")
    
    # Generate predictions for confusion matrix
    y_pred = np.argmax(model.predict(x_test), axis=1)
    
    # Save model
    model.save(f"cifar10_{optimizer_name}.h5")
    
    return {
        'history': history.history,
        'accuracy': test_accuracy,
        'training_time': training_time,
        'predictions': y_pred
    }

# Plot training history comparison
def plot_training_curves(results):
    plt.figure(figsize=(20, 10))
    
    # Plot training accuracy
    plt.subplot(2, 2, 1)
    for optimizer_name, result in results.items():
        plt.plot(result['history']['accuracy'], label=optimizer_name)
    
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(2, 2, 2)
    for optimizer_name, result in results.items():
        plt.plot(result['history']['val_accuracy'], label=optimizer_name)
    
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot training loss
    plt.subplot(2, 2, 3)
    for optimizer_name, result in results.items():
        plt.plot(result['history']['loss'], label=optimizer_name)
    
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Plot validation loss
    plt.subplot(2, 2, 4)
    for optimizer_name, result in results.items():
        plt.plot(result['history']['val_loss'], label=optimizer_name)
    
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_curves.png')
    plt.close()

# Plot final accuracy and training time comparison
def plot_performance_comparison(results):
    # Extract data for plotting
    optimizer_names = list(results.keys())
    accuracies = [results[name]['accuracy'] * 100 for name in optimizer_names]
    training_times = [results[name]['training_time'] for name in optimizer_names]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy comparison
    ax1.bar(optimizer_names, accuracies, color='skyblue')
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim([0, 100])
    
    # Add value labels
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 1, f"{v:.2f}%", ha='center')
    
    # Plot training time comparison
    ax2.bar(optimizer_names, training_times, color='salmon')
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    
    # Add value labels
    for i, v in enumerate(training_times):
        ax2.text(i, v + 5, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig('optimizer_performance_comparison.png')
    plt.close()
    
    # Create a summary DataFrame and save to CSV
    summary_df = pd.DataFrame({
        'Optimizer': optimizer_names,
        'Accuracy (%)': accuracies,
        'Training Time (s)': training_times
    })
    
    summary_df.to_csv('optimizer_comparison_results.csv', index=False)
    print("\nResults saved to 'optimizer_comparison_results.csv'")

# Plot confusion matrix for the best optimizer
def plot_confusion_matrix(y_true, y_pred, optimizer_name):
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {optimizer_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{optimizer_name}.png')
    plt.close()

def main():
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Get optimizers to test
    optimizers = get_optimizers(learning_rate=0.01)
    
    # Train and evaluate with each optimizer
    results = {}
    for name, optimizer in optimizers.items():
        result = train_and_evaluate(name, optimizer, x_train, y_train, x_test, y_test, epochs=15, batch_size=64)
        results[name] = result
    
    # Plot training curves
    plot_training_curves(results)
    
    # Plot performance comparison
    plot_performance_comparison(results)
    
    # Find best optimizer and plot its confusion matrix
    best_optimizer = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    print(f"\nBest optimizer: {best_optimizer} with accuracy {results[best_optimizer]['accuracy']*100:.2f}%")
    
    plot_confusion_matrix(y_test, results[best_optimizer]['predictions'], best_optimizer)

if __name__ == "__main__":
    main()
```

 
