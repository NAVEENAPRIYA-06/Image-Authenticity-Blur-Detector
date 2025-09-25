import sys
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Add the parent directory to the system path to find the 'utils' folder
sys.path.append(os.path.dirname(__file__))

# Import the model builder function
from utils.authenticity_classifier import build_cnn_model

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset', 'authenticity')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create the models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Image dimensions and batch size
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32

# Data Generators with Data Augmentation
# This helps the model generalize better by creating variations of your training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data from the directories
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build the model using our function
model = build_cnn_model((IMG_HEIGHT, IMG_WIDTH, 3))

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Save the best model during training
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'authenticity_model.h5'),
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Train the model
# The training will take a while, be patient!
history = model.fit(
    train_generator,
    epochs=30,  # You can increase this for better accuracy
    validation_data=validation_generator,
    callbacks=[checkpoint]
)

print("Training complete. Model saved to 'models/authenticity_model.h5'")