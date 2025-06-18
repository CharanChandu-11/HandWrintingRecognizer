import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class MNISTDigitRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the MNIST dataset"""
        print("Loading MNIST dataset...")
        
        # Load the MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to add channel dimension (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical one-hot encoding
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
    def build_cnn_model(self):
        """Build a Convolutional Neural Network for digit recognition"""
        print("Building CNN model...")
        
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Display model architecture
        print("\nModel Architecture:")
        model.summary()
        
    def train_model(self, epochs=15, batch_size=128):
        """Train the CNN model"""
        print(f"\nTraining model for {epochs} epochs...")
        
        # Define callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\nEvaluating model...")
        
        # Evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        return y_pred_classes, y_true_classes
        
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self, y_pred, y_true):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
    def visualize_predictions(self, num_samples=10):
        """Visualize model predictions on test samples"""
        # Get random samples
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        # Make predictions
        predictions = self.model.predict(self.x_test[indices])
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(self.y_test[indices], axis=1)
        
        # Plot samples with predictions
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Display image
            axes[i].imshow(self.x_test[idx].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}')
            axes[i].axis('off')
            
            # Color the title based on correctness
            if true_classes[i] == predicted_classes[i]:
                axes[i].set_title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}', 
                                color='green')
            else:
                axes[i].set_title(f'True: {true_classes[i]}, Pred: {predicted_classes[i]}', 
                                color='red')
        
        plt.tight_layout()
        plt.show()
        
    def predict_digit(self, image):
        """Predict a single digit from an image"""
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return None
            
        # Preprocess the image
        if image.shape != (28, 28, 1):
            image = image.reshape(1, 28, 28, 1)
        else:
            image = image.reshape(1, 28, 28, 1)
            
        # Normalize if needed
        if image.max() > 1:
            image = image / 255.0
            
        # Make prediction
        prediction = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_class, confidence
        
    def save_model(self, filepath='mnist_cnn_model.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='mnist_cnn_model.h5'):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

def main():
    """Main function to run the MNIST digit recognizer"""
    print("=== MNIST Handwritten Digit Recognizer ===")
    print("Using Convolutional Neural Network (CNN)\n")
    
    # Initialize the recognizer
    recognizer = MNISTDigitRecognizer()
    
    # Load and preprocess data
    recognizer.load_and_preprocess_data()
    
    # Build the CNN model
    recognizer.build_cnn_model()
    
    # Train the model
    recognizer.train_model(epochs=15, batch_size=128)
    
    # Evaluate the model
    y_pred, y_true = recognizer.evaluate_model()
    
    # Visualize results
    recognizer.plot_training_history()
    recognizer.plot_confusion_matrix(y_pred, y_true)
    recognizer.visualize_predictions(num_samples=10)
    
    # Save the model
    recognizer.save_model('mnist_cnn_model.h5')
    
    print("\n=== Training Complete! ===")
    print("Your MNIST digit recognizer is ready to use!")

if __name__ == "__main__":
    main()

# Example usage for making predictions on new images:
"""
# Load a trained model
recognizer = MNISTDigitRecognizer()
recognizer.load_model('mnist_cnn_model.h5')

# Load MNIST test data for demonstration
(_, _), (x_test, _) = keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0

# Predict a single digit
sample_image = x_test[0]  # First test image
predicted_digit, confidence = recognizer.predict_digit(sample_image)
print(f"Predicted digit: {predicted_digit} (Confidence: {confidence:.2f})")
"""