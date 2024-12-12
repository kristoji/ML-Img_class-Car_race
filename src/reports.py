import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_loss(history, report_dir):
    """Plot and save training loss."""
    os.makedirs(report_dir, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'{report_dir}/loss_plot.png')

def plot_accuracy(history, report_dir):
    """Plot and save training accuracy."""
    os.makedirs(report_dir, exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'{report_dir}/accuracy_plot.png')

def plot_f1_score(f1_callback, report_dir):
    """Plot and save F1 scores."""
    os.makedirs(report_dir, exist_ok=True)

    # Plot F1 scores
    plt.figure(figsize=(8, 6))
    plt.plot(f1_callback.training_f1_scores, label='Training F1 Score')
    plt.plot(f1_callback.validation_f1_scores, label='Validation F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid()
    plt.savefig(f'{report_dir}/f1_score_plot.png')

def plot_confusion_matrix(y_test, y_pred, num_classes, report_dir):
    """Plot and save confusion matrix."""
    os.makedirs(report_dir, exist_ok=True)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=[f'Class {i}' for i in range(num_classes)], yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{report_dir}/confusion_matrix.png')

def save_classification_report(y_test, y_pred, num_classes, report_dir, verbose=True):
    """Save the classification report as a text file."""
    os.makedirs(report_dir, exist_ok=True)

    report = classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(num_classes)])
    if verbose:
        print(report)
    with open(f'{report_dir}/classification_report.txt', 'w') as f:
        f.write(report)

def save_summary(model, report_dir, verbose=True):
    """Save the model and its summary."""
    os.makedirs(report_dir, exist_ok=True)

    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary_text = "\n".join(model_summary)
    if verbose:
        print(model_summary_text)

    # Save model summary
    with open(f'{report_dir}/model_summary.txt', 'w') as f:
        f.write(model_summary_text)

