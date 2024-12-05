import os
import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
import pandas as pd

# Reproducibility
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed()

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells

# Load the CSV files
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

# Extract labels and images from the DataFrame
train_labels = train_df.pop('label').values
train_images = train_df.values

# Normalize the image data
train_images = train_images / 255.0

# Reshape the images to their original shape (assuming 28x28 pixels)
train_images = train_images.reshape(-1, 28, 28, 1)

# Convert to TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

# Split the dataset into training and validation sets
val_size = int(0.2 * len(train_images))
val_dataset = train_dataset.take(val_size)
train_dataset = train_dataset.skip(val_size)

# Batch and shuffle the datasets
batch_size = 64
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    train_dataset
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    val_dataset
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
import tensorflow as tf
import matplotlib.pyplot as plt
import learntools.computer_vision.visiontools as visiontools


plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
import tensorflow as tf
from tensorflow.keras import layers, regularizers
# Set random seed for reproducibility
tf.random.set_seed(31415)

# Define the CNN model with refined architecture and regularization
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),  # Input layer
    layers.RandomRotation(factor=0.1),  # Data augmentation
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Data augmentation
    layers.RandomFlip(mode='horizontal'),  # Data augmentation
    layers.Resizing(28, 28),  # Ensure fixed size after random width and height

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # First Conv layer
    layers.BatchNormalization(),  # Batch normalization
    layers.MaxPooling2D((2, 2)),  # First MaxPooling layer
    layers.Dropout(0.3),  # Dropout to prevent overfitting

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # Second Conv layer
    layers.BatchNormalization(),  # Batch normalization
    layers.MaxPooling2D((2, 2)),  # Second MaxPooling layer
    layers.Dropout(0.3),  # Dropout to prevent overfitting

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # Third Conv layer
    layers.BatchNormalization(),  # Batch normalization
    layers.MaxPooling2D((2, 2)),  # Third MaxPooling layer
    layers.Dropout(0.4),  # Dropout to prevent overfitting

    layers.Flatten(),  # Flatten layer

    layers.Dense(128, activation='relu'),  # Dense layer
    layers.BatchNormalization(),  # Batch normalization
    layers.Dropout(0.5),  # Dropout to prevent overfitting

    layers.Dense(10, activation='softmax')  # Output layer assuming 10 classes
])

model.summary()

# Compile the model with a learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.9
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

# Train the model
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
    callbacks=[early_stopping]
)
import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
## Normalize the test images
test_images = test_df.values / 255.0

# Reshape the test images to their original shape (28x28 pixels)
test_images = test_images.reshape(-1, 28, 28, 1)

# Convert to TensorFlow dataset
test_dataset = tf.data.Dataset.from_tensor_slices(test_images)

# Batch the test dataset
test_dataset = test_dataset.batch(batch_size)

# Make predictions
predictions = model.predict(test_dataset)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Create a DataFrame with the submission data
submission_df = pd.DataFrame({
    'ImageId': np.arange(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})

# Save the DataFrame to a CSV file
submission_file = 'submission.csv'
submission_df.to_csv(submission_file, index=False)