import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# PARAMETERS
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
DATASET_PATH = 'dataset'

# DATASET LOADING
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Show class index mapping
print("Class indices:", train_data.class_indices)  # Expect {'closed': 0, 'open': 1}

# MODEL DEFINITION
base_model = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                         include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # Dropout to prevent overfitting
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# MODEL TRAINING
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# EVALUATE ON VALIDATION
loss, acc = model.evaluate(val_data)
print(f" Final Validation Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")

# SAVE MODEL
model.save("eye_state_classifier.keras")
print("Model saved as eye_state_classifier.keras")
