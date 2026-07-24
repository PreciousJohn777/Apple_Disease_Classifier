# =====================================================
# APPLE DISEASE CLASSIFICATION USING CNN
# CE1 MINI PROJECT
# =====================================================

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Path to dataset
dataset_path = "dataset/Apple"

# Create ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

# Validation data
validation_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

print("\nDataset loaded successfully!")
print("Classes:", train_data.class_indices)