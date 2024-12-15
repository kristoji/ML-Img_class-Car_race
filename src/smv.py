import numpy as np
import tensorflow as tf
from sklearn import svm
import matplotlib.pyplot as plt
from keras.applications import VGG16
from sklearn.metrics import accuracy_score


def extract_features(dataset, model, image_size=(96, 96)):
    features = []
    labels = []
    for images, lbls in dataset:
        feats = model.predict(images, verbose=0)
        features.append(feats)
        labels.append(lbls)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    features = np.mean(features, axis=(1, 2))  # Shape: (batch_size, feature_dim)
    
    return features, labels

def create_feature_extractor(input_shape=(96, 96, 3), init_weights='imagenet'):
    base_model = VGG16(include_top=False, weights=init_weights, input_shape=input_shape)
    base_model.trainable = False  # Freeze the layers
    return base_model

def load_data():
    trainingset = '../Dataset/train'
    testset = '../Dataset/test'
    
    image_size = (96, 96)
    
    vgg16feat_model = create_feature_extractor(input_shape=image_size+(3,))
    
    train_flow = tf.keras.preprocessing.image_dataset_from_directory(
        trainingset,
        image_size=image_size,
        label_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    test_flow = tf.keras.preprocessing.image_dataset_from_directory(
        testset,
        image_size=image_size,
        label_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    x_train_feat, y_train = extract_features(train_flow, vgg16feat_model)
    x_test_feat, y_test = extract_features(test_flow, vgg16feat_model)
    
    return x_train_feat, y_train, x_test_feat, y_test

def train_svm(x_train_feat, y_train, x_test_feat, y_test, kernel='sigmoid', C=0.01, degree=3):
    
    clf = svm.SVC(kernel=kernel, C=C, degree=degree)
    clf.fit(x_train_feat, np.argmax(y_train, axis=1))
    
    y_pred = clf.predict(x_test_feat)
    
    accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    print(f"Test set accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    x_train_feat, y_train, x_test_feat, y_test = load_data()

    kernels = []
    C_values = []
    accuracies = []

    for kernel in ['rbf', 'sigmoid',"poly"]:
        for C in [0.001, 0.0001, 0.01, 0.1, 1]:
            for degree in [3, 5, 9, 15]:    
                
                print(f"Kernel: {kernel}, Degree: {degree}, C: {C}")
                accuracy = train_svm(x_train_feat, y_train, x_test_feat, y_test,
                                kernel=kernel, C=C, degree=degree)
                print(f"Accuracy: {accuracy * 100:.2f}%")
                C_values.append(C)
                accuracies.append(accuracy * 100)
                if kernel == "poly":
                    kernels.append(kernel + f" deg{degree}")
                else:
                    kernels.append(kernel)
                    break

    plot_results(kernels, C_values, accuracies)
    return

def plot_results(kernels, C_values, accuracies):
    plt.figure(figsize=(10, 6))
    for kernel in set(kernels):
        kernel_accuracies = [accuracies[i] for i in range(len(kernels)) if kernels[i] == kernel]
        kernel_C_values = [C_values[i] for i in range(len(kernels)) if kernels[i] == kernel]
        plt.plot(kernel_C_values, kernel_accuracies, marker='o', label=f"Kernel: {kernel}")

    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('Test Set Accuracy (%)')
    plt.title('SVM Hyperparameter Search: Kernel and C')
    plt.legend()
    plt.grid(True)
    plt.savefig('svm_hyperparameter_search.png')
    plt.close()

if __name__ == '__main__':  
    main()
