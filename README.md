# MNIST CNN Model Training and Evaluation
This repository contains a Python script that implements, trains, and evaluates a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset using TensorFlow and Keras.

## Features
Data Loading and Preprocessing:
Loads the MNIST dataset, reshapes images to include the grayscale channel, and normalizes pixel values to the [0, 1] range.

## Model Architecture:
Builds a sequential CNN with three convolutional layers, max-pooling, and fully connected dense layers, ending with a softmax output for 10 classes (digits 0-9).

## Training:
Compiles the model using the Adam optimizer and sparse categorical cross-entropy loss, and trains it for 5 epochs with validation on the test set.

## Evaluation:
Predicts labels on the test set and computes:

Confusion matrix (displayed as a normalized heatmap using seaborn)

Key classification metrics including Precision, Recall, F1-score, Accuracy, and Specificity, calculated manually from the confusion matrix.

## Visualization:
Provides a clear visualization of the normalized confusion matrix for better interpretability of model performance.

