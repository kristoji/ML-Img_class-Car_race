import os
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
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

def read_img(path, crop_top=False, crop_bottom=False, apply_remove_green=False, resize=(96,96), grayscale=False):
    if resize[0] != resize[1]:
        print("[E] Resize dimensions must be equal")
        exit()
    
    if grayscale:
        img = Image.open(path).convert('L')
    else:
        img = Image.open(path).convert('RGB')
    img = img.resize(resize)

    crop = resize[0] // 8
    
    if crop_top and crop_bottom:
        print("[E] Cannot crop top and bottom at the same time")

    if crop_top:
        img = img.crop((0, resize[1]-crop, resize[0], resize[1]))
    elif crop_bottom:
        # img = img.crop((0, resize[1] - crop, resize[0], resize[1]))
        img = img.crop((0,0,resize[0],resize[1]-crop))
    
    img = np.array(img)
    if apply_remove_green:
        if grayscale:
            print("[E] Cannot remove green from grayscale image")
            exit()
        img = remove_green(img)
    return img

def load_data(path):
    X_train, y_train, X_test, y_test = [], [], [], []
    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('.png'):
                X_train.append(read_img(os.path.join(root, file), resize=(32,32), grayscale=True))
                y_train.append(int(root.split('/')[-1]))
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.png'):
                X_test.append(read_img(os.path.join(root, file), resize=(32,32), grayscale=True))
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

def build_leNet_model(input_shape, num_classes, lr=0.001):

    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = optimizers.Adam(learning_rate=lr)  # try other optimizers and hyper-parameters

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
        y_train_true = y_train
        train_f1 = f1_score(y_train_true, y_train_pred, average='weighted')
        self.training_f1_scores.append(train_f1)

        # Validation data
        X_val, y_val = self.validation_data
        y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        y_val_true = y_val
        val_f1 = f1_score(y_val_true, y_val_pred, average='weighted')
        self.validation_f1_scores.append(val_f1)

        print(f"Epoch {epoch + 1}: Training F1 Score = {train_f1:.4f}, Validation F1 Score = {val_f1:.4f}")

if __name__ == '__main__':
    N=6

    X_train, y_train, X_test, y_test = load_data('../Dataset')

    # Normalize image data to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(-1, 32, 32, 1)
    X_test = X_test.reshape(-1, 32, 32, 1)


    # Convert labels to categorical (one-hot encoding)
    num_classes = 5
    # y_train = to_categorical(y_train, num_classes)
    # y_test = to_categorical(y_test, num_classes)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # exit()

    # Build the CNN model
    # input_shape = (96, 96, 3)
    input_shape = (32,32,1)
    # model = build_cnn_model(input_shape, num_classes)
    model = build_leNet_model(input_shape, num_classes)


    # Print the model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_text = "\n".join(model_summary)
    print(model_summary_text)

    # Initialize F1ScoreCallback
    f1_callback = F1ScoreCallback(training_data=(X_train, y_train), validation_data=(X_test, y_test))

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        callbacks=[f1_callback],
        batch_size=256
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
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'../Report/{N}/accuracy_plot.png')

    # Plot F1 score over epochs
    plt.figure(figsize=(8, 6))
    plt.plot(f1_callback.training_f1_scores, label='Training F1 Score')
    plt.plot(f1_callback.validation_f1_scores, label='Validation F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    f1_plot_path = f'../Report/{N}/f1_score_plot.png'
    plt.savefig(f1_plot_path)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = y_test

    # Classification report
    report = classification_report(y_test_classes, y_pred_classes, target_names=[f'Class {i}' for i in range(num_classes)])
    print("Classification Report:")
    print(report)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[f'Class {i}' for i in range(num_classes)], yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save the confusion matrix to a file
    conf_matrix_path = f"../Report/{N}/confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    plt.close()

    # Save the classification report as a text file
    with open(f'../Report/{N}/classification_report.txt', 'w') as f:
        f.write(report)

    with open(f'../Report/{N}/model_summary.txt', 'w') as f:
        f.write(model_summary_text)

    # Save the model
    model.save(f'../Report/{N}/cnn_model.keras')
    print("Model saved as cnn_model.keras")
