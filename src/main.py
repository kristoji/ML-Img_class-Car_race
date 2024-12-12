import os
import parse
import models
import reports
import numpy as np
import tensorflow as tf

# TF_CPP_MIN_LOG_LEVEL=2 python main.py
if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)

    REPORT=8
    OVERWRITE = True

    LENET = True
    BATCH_SIZE = 32
    EPOCHS = 50


    NUM_CLASSES = 5
    data_path = '../Dataset'
    if LENET:
        resize = (32,32)
        grayscale = True
        apply_remove_green = False
        crop_bottom = True
    else:
        resize = (96,96)
        grayscale = False
        apply_remove_green = True
        crop_bottom = True

    # Create the report directory
    report_dir = f'../Report/{REPORT}'
    os.makedirs(report_dir, exist_ok=OVERWRITE)

    # Load and preprocess the data
    X_train, y_train, X_test, y_test = parse.preprocess_data(data_path, resize=resize, grayscale=grayscale, apply_remove_green=apply_remove_green, crop_bottom=crop_bottom)

    # exit()

    # Compile the model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    if LENET:
        model = models.build_leNet_model(input_shape, NUM_CLASSES)
    else:
        model = models.build_cnn_model(input_shape, NUM_CLASSES)

    # Print the model summary
    reports.save_summary(model, report_dir)

    # Initialize F1ScoreCallback
    f1_callback = models.F1ScoreCallback(
        training_data=(X_train, y_train), 
        validation_data=(X_test, y_test)
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=[f1_callback],
        batch_size=BATCH_SIZE
    )

    # Plot and save the training loss
    reports.plot_loss(history, report_dir)

    # Plot and save the training accuracy
    reports.plot_accuracy(history, report_dir)

    # Plot and save the F1 scores
    reports.plot_f1_score(f1_callback, report_dir)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = y_test

    # Classification report
    reports.save_classification_report(y_test_classes, y_pred_classes, NUM_CLASSES, report_dir)

    # Confusion matrix
    reports.plot_confusion_matrix(y_test_classes, y_pred_classes, NUM_CLASSES, report_dir)

    # Save the model
    model.save(f'{report_dir}/cnn_model.keras')
