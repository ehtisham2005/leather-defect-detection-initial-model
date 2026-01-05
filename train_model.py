import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- CONFIGURATION ---
TRAIN_DIR = os.path.join('dataset', 'train')
VAL_DIR = os.path.join('dataset', 'valid')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 1. LOAD DATA ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, shuffle=True, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, shuffle=False, image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 2. BUILD MODEL (THE FIX) ---
base_model = tf.keras.applications.ResNet50(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
base_model.trainable = False 

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

# --- FIX: Wrap preprocess_input in a Lambda layer ---
# This prevents the "Unknown layer: GetItem" error
x = tf.keras.layers.Lambda(preprocess_input, name='preprocessing')(inputs)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. TRAIN & SAVE ---
print("\nStarting Training (Final Fix)...")
epochs = 15 
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# We save in the newer .keras format to be extra safe, but .h5 is fine too.
# Let's stick to .h5 to match your app code.
model.save('leather_defect_model.h5')
print("\nSUCCESS! Model saved as 'leather_defect_model.h5'")