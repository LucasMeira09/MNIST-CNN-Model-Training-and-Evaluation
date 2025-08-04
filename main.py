# Import necessary libraries
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd

# Directory for TensorBoard logs (if used)
logdir = 'log'

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape images to add channel dimension (grayscale)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# List of classes (digits 0 to 9)
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Define and train the CNN model on MNIST
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes for MNIST

# Callback for TensorBoard (optional)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x=train_images,
          y=train_labels,
          epochs=5,
          validation_data=(test_images, test_labels))

# Predictions on the test set
y_true = test_labels
y_pred = model.predict(test_images)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert to predicted labels

# Confusion matrix
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred_labels).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

# Create a DataFrame for visualization with seaborn
con_mat_df = pd.DataFrame(con_mat_norm,
                          index=classes,
                          columns=classes)

# Display the normalized confusion matrix
figures = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Manual calculation of global metrics
n_classes = con_mat.shape[0]
TP = np.diag(con_mat)  # True positives
FP = np.sum(con_mat, axis=0) - TP  # False positives
FN = np.sum(con_mat, axis=1) - TP  # False negatives
TN = []  # True negatives per class

# Calculate TN for each class
total = np.sum(con_mat)
for i in range(n_classes):
    tn_i = total - (TP[i] + FP[i] + FN[i])
    TN.append(tn_i)

# Sum of all metrics
TP_total = np.sum(TP)
FP_total = np.sum(FP)
FN_total = np.sum(FN)
TN_total = np.sum(TN)

# Compute overall metrics
precision = TP_total / (TP_total + FP_total)
recall = TP_total / (TP_total + FN_total)
F1 = 2 * (precision * recall) / (precision + recall)
accuracy = (TP_total + TN_total) / (TP_total + FN_total + TN_total + FP_total)
specificity = TN_total / (TN_total + FP_total)

# Display the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {F1}")
print(f"Accuracy: {accuracy}")
print(f"Specificity: {specificity}")
