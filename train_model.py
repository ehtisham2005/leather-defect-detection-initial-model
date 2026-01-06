import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import itertools
from sklearn.metrics import f1_score, confusion_matrix, classification_report

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

# --- 4. METRICS: F1 SCORE & CONFUSION MATRIX ---
# Build true / predicted arrays from the validation dataset
print("\nComputing predictions on validation set for metrics...")
y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# F1 score (weighted for multi-class)
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 score (weighted): {f1:.4f}")

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)
thresh = cm.max() / 2. if cm.size else 0
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Ensure static directory exists and save figure
os.makedirs('static', exist_ok=True)
cm_path = os.path.join('static', 'confusion_matrix.png')
plt.savefig(cm_path)
print(f"Confusion matrix saved to '{cm_path}'")