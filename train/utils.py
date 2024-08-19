# FraudDetectionHybrid/utils.py

import pickle
import matplotlib.pyplot as plt

def save_model(model, file_path):
    """Saves the trained model to a file."""
    model.save(file_path)

def load_model(file_path):
    """Loads a model from a file."""
    from tensorflow.keras.models import load_model
    return load_model(file_path)

def plot_metrics(history):
    """Plots training metrics such as loss and accuracy."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
