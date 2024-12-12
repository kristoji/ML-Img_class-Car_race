from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from sklearn.metrics import f1_score
from tensorflow.keras.callbacks import Callback
import numpy as np


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
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
        # y_train_true = y_train
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        self.training_f1_scores.append(train_f1)

        # Validation data
        X_val, y_val = self.validation_data
        y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        # y_val_true = y_val
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        self.validation_f1_scores.append(val_f1)

        # print(f"Epoch {epoch + 1}: Training F1 Score = {train_f1:.4f}, Validation F1 Score = {val_f1:.4f}")
