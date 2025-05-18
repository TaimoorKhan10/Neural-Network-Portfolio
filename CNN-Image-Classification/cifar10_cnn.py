
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
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check for GPU availability
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU Devices: ", tf.config.list_physical_devices('GPU'))

# Create output directory for saved models and visualizations
if not os.path.exists('output'):
    os.makedirs('output')

def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Print dataset information
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # One-hot encode the labels
    y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
    
    # Set up data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(x_train)
    
    return (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot), datagen

def display_sample_images(x_train, y_train, class_names):
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
    plt.savefig('output/cifar10_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train_onehot, x_test, y_test_onehot, datagen):
    # Define callbacks for training
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
            'output/best_cifar10_model.h5',
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
    epochs = 50
    
    history = model.fit(
        datagen.flow(x_train, y_train_onehot, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_onehot),
        callbacks=callbacks
    )
    
    return history

def plot_training_history(history):
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
    plt.savefig('output/cifar10_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, x_test, y_test_onehot, class_names):
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test_onehot, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_onehot, axis=1)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                  target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("Classification Report:")
    print(report_df)
    report_df.to_csv('output/classification_report.csv')
    
    return y_pred_classes, y_true_classes

def visualize_misclassifications(x_test, y_pred_classes, y_true_classes, class_names):
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
    plt.savefig('output/misclassified_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_feature_maps(model, x_test, y_test, class_names):
    # Create a model that returns feature maps
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][:2]
    feature_map_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get a correctly classified test image
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    correctly_classified = np.where(y_pred_classes == y_test.flatten())[0]
    img_index = correctly_classified[0]  # Take the first correctly classified example
    
    # Display the test image
    plt.figure(figsize=(4, 4))
    plt.imshow(x_test[img_index])
    plt.title(f"Class: {class_names[y_test[img_index][0]]}")
    plt.axis('off')
    plt.savefig('output/feature_map_input.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Get feature maps
    feature_maps = feature_map_model.predict(np.expand_dims(x_test[img_index], axis=0))
    
    # Plot feature maps
    for i, feature_map in enumerate(feature_maps):
        n_features = min(16, feature_map.shape[-1])
        size = int(np.ceil(np.sqrt(n_features)))
        
        plt.figure(figsize=(12, 12))
        plt.suptitle(f'Feature maps from convolutional layer {i+1}', fontsize=16)
        
        for j in range(n_features):
            plt.subplot(size, size, j+1)
            plt.imshow(feature_map[0, :, :, j], cmap='viridis')
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'output/feature_maps_layer{i+1}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load and preprocess data
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot), datagen = load_and_preprocess_data()
    
    # Display sample images
    display_sample_images(x_train, y_train, class_names)
    
    # Build the CNN model
    model = build_cnn_model()
    model.summary()
    
    # Train the model
    history = train_model(model, x_train, y_train_onehot, x_test, y_test_onehot, datagen)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    y_pred_classes, y_true_classes = evaluate_model(model, x_test, y_test_onehot, class_names)
    
    # Visualize misclassifications
    visualize_misclassifications(x_test, y_pred_classes, y_true_classes, class_names)
    
    # Visualize feature maps
    visualize_feature_maps(model, x_test, y_test, class_names)
    
    # Save the final model
    model.save('output/final_cifar10_cnn_model.h5')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
```

