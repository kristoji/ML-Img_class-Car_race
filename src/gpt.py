import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback

def remove_green(img, threshold=50):
    target_green = [0, 255, 0]
    green_mask = (img[:, :, 1] > img[:, :, 0] + threshold) & \
                 (img[:, :, 1] > img[:, :, 2] + threshold) & \
                 (img[:, :, 0] < 200) & (img[:, :, 2] < 200)
    modified_array = img.copy()
    modified_array[green_mask] = target_green
    return modified_array

def read_img(path, crop_top=False, crop_bottom=False, apply_remove_green=False):
    img = Image.open(path).convert('RGB')
    img = img.resize((96, 96))
    if crop_top:
        img = img.crop((0, 84, 96, 96))
    elif crop_bottom:
        img = img.crop((0, 0, 96, 12))
    img = np.array(img)
    if apply_remove_green:
        img = remove_green(img)
    return img

def load_data(path):
    X_train, y_train, X_test, y_test = [], [], [], []
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.png'):
                X_train.append(read_img(os.path.join(root, file)))
                y_train.append(int(root.split('/')[-1]))
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.png'):
                X_test.append(read_img(os.path.join(root, file)))
                y_test.append(int(root.split('/')[-1]))
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class F1ScoreCallback(Callback):
    def __init__(self, training_data, validation_data):
        super().__init__()
        self.training_data = training_data
        self.validation_data = validation_data
        self.training_f1_scores = []
        self.validation_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        # Training data
        X_train, y_train = self.training_data
        y_train_pred = np.argmax(self.model.predict(X_train), axis=1)
        y_train_true = np.argmax(y_train, axis=1)
        train_f1 = f1_score(y_train_true, y_train_pred, average='macro')
        self.training_f1_scores.append(train_f1)

        # Validation data
        X_val, y_val = self.validation_data
        y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        y_val_true = np.argmax(y_val, axis=1)
        val_f1 = f1_score(y_val_true, y_val_pred, average='macro')
        self.validation_f1_scores.append(val_f1)

        print(f"Epoch {epoch + 1}: Training F1 Score = {train_f1:.4f}, Validation F1 Score = {val_f1:.4f}")

if __name__ == '__main__':
    # Load and preprocess the data
    base_path = '../Dataset'  # Update this to the correct dataset path
    X_train, y_train, X_test, y_test = load_data(base_path)

    # Normalize image data to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert labels to categorical (one-hot encoding)
    num_classes = 5
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build the CNN model
    input_shape = (96, 96, 3)
    model = build_cnn_model(input_shape, num_classes)

    # Initialize F1ScoreCallback
    f1_callback = F1ScoreCallback(training_data=(X_train, y_train), validation_data=(X_val, y_val))

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[f1_callback]
    )

    # Plot training and validation loss over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    loss_plot_path = './loss_plot.png'
    plt.savefig(loss_plot_path)

    # Plot F1 score over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(f1_callback.training_f1_scores, label='Training F1 Score', marker='o')
    plt.plot(f1_callback.validation_f1_scores, label='Validation F1 Score', marker='o')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    f1_plot_path = './f1_score_plot.png'
    plt.savefig(f1_plot_path)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Classification report
    report = classification_report(y_test_classes, y_pred_classes, target_names=[f'Class {i}' for i in range(num_classes)])
    print("Classification Report:")
    print(report)

    # Save the classification report
    report_path = './classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write(report)

    # Save the model
    model_path = './cnn_model.keras'
    model.save(model_path)
    print("Model saved as cnn_model.keras")
