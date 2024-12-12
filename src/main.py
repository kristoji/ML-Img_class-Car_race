import os
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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

def load_data(path, crop_top=False, crop_bottom=False, apply_remove_green=False):
    X_train, y_train, X_test, y_test = [], [], [], []
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.png'):
                X_train.append(read_img(os.path.join(root, file), crop_bottom=crop_bottom, crop_top=crop_top, apply_remove_green=apply_remove_green))
                y_train.append(int(root.split('/')[-1]))
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.png'):
                X_test.append(read_img(os.path.join(root, file), crop_bottom=crop_bottom, crop_top=crop_top, apply_remove_green=apply_remove_green))
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

if __name__ == '__main__':
    N=1

    X_train, y_train, X_test, y_test = load_data('../Dataset')

    # Normalize image data to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Convert labels to categorical (one-hot encoding)
    num_classes = 5
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Build the CNN model
    input_shape = (96, 96, 3)
    model = build_cnn_model(input_shape, num_classes)


    # Print the model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_text = "\n".join(model_summary)
    print(model_summary_text)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32
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
    plt.savefig(f'../Report/{N}/loss_plot.png')

    # Plot training and validation loss over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'../Report/{N}/accuracy_plot.png')

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

    # Save the classification report as a text file
    with open(f'../Report/{N}/classification_report.txt', 'w') as f:
        f.write(report)

    with open(f'../Report/{N}/model_summary.txt', 'w') as f:
        f.write(model_summary_text)

    # Save the model
    model.save(f'../Report/{N}/cnn_model.keras')
    print("Model saved as cnn_model.keras")
