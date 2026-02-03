"""
Logo Classification CNN Training
================================
A robust CNN model for classifying logo images.
Uses transfer learning with MobileNetV2 for better performance with small datasets.
"""

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# =====================================================
# 1. CONFIGURATION
# =====================================================
DATA_DIR = pathlib.Path("data/logos_small")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 16
SEED = 42
EPOCHS_STAGE1 = 30  # Frozen base
EPOCHS_STAGE2 = 15  # Fine-tuning

# =====================================================
# 2. DATA LOADING & PREPROCESSING
# =====================================================
print("=" * 60)
print("Loading Training and Validation Datasets...")
print("=" * 60)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nClasses: {class_names}")
print(f"Number of classes: {num_classes}")

# Count images per class
print("\nTraining set distribution:")
for cls in os.listdir(TRAIN_DIR):
    cls_path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        print(f"  {cls}: {count} images")

print("\nValidation set distribution:")
for cls in os.listdir(VAL_DIR):
    cls_path = os.path.join(VAL_DIR, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        print(f"  {cls}: {count} images")

# Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =====================================================
# 3. DATA AUGMENTATION
# =====================================================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.2, 0.2),
], name="data_augmentation")

# =====================================================
# 4. MODEL BUILDING (Transfer Learning with MobileNetV2)
# =====================================================
print("\n" + "=" * 60)
print("Building CNN Model with Transfer Learning (MobileNetV2)...")
print("=" * 60)

# Load pre-trained MobileNetV2 (without top classification layer)
base_model = MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet"
)

# Freeze the base model initially
base_model.trainable = False

# Build the complete model
inputs = keras.Input(shape=IMG_SHAPE, name="input_layer")
x = data_augmentation(inputs)
x = preprocess_input(x)  # MobileNetV2 preprocessing
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
x = layers.BatchNormalization(name="batch_norm")(x)
x = layers.Dropout(0.5, name="dropout_1")(x)
x = layers.Dense(256, activation="relu", name="dense_1")(x)
x = layers.Dropout(0.3, name="dropout_2")(x)
outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

model = keras.Model(inputs, outputs, name="logo_classifier")
model.summary()

# =====================================================
# 5. STAGE 1: TRAIN WITH FROZEN BASE
# =====================================================
print("\n" + "=" * 60)
print("STAGE 1: Training with Frozen Base Model...")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_logo_model_stage1.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# =====================================================
# 6. STAGE 2: FINE-TUNING
# =====================================================
print("\n" + "=" * 60)
print("STAGE 2: Fine-tuning the Model...")
print("=" * 60)

# Unfreeze the top layers of the base model
base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Total layers in base model: {len(base_model.layers)}")
print(f"Trainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}")

# Re-compile with a lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint2 = ModelCheckpoint(
    "best_logo_model_finetuned.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=[early_stop, reduce_lr, checkpoint2]
)

# =====================================================
# 7. SAVE FINAL MODEL
# =====================================================
model.save("logo_classifier_final.keras")
print("\n✓ Model saved as 'logo_classifier_final.keras'")

# =====================================================
# 8. TRAINING HISTORY VISUALIZATION
# =====================================================
def plot_history(history1, history2=None):
    """Plot training history with optional two-stage visualization."""
    if history2:
        # Combine histories
        acc = history1.history["accuracy"] + history2.history["accuracy"]
        val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
        loss = history1.history["loss"] + history2.history["loss"]
        val_loss = history1.history["val_loss"] + history2.history["val_loss"]
        stage1_end = len(history1.history["accuracy"])
    else:
        acc = history1.history["accuracy"]
        val_acc = history1.history["val_accuracy"]
        loss = history1.history["loss"]
        val_loss = history1.history["val_loss"]
        stage1_end = None

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    axes[0].plot(epochs_range, acc, 'b-', label="Training Accuracy", linewidth=2)
    axes[0].plot(epochs_range, val_acc, 'r-', label="Validation Accuracy", linewidth=2)
    if stage1_end:
        axes[0].axvline(x=stage1_end, color='gray', linestyle='--', label='Fine-tuning Start')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss Plot
    axes[1].plot(epochs_range, loss, 'b-', label="Training Loss", linewidth=2)
    axes[1].plot(epochs_range, val_loss, 'r-', label="Validation Loss", linewidth=2)
    if stage1_end:
        axes[1].axvline(x=stage1_end, color='gray', linestyle='--', label='Fine-tuning Start')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("✓ Training history saved as 'training_history.png'")

plot_history(history1, history2)

# =====================================================
# 9. MODEL EVALUATION
# =====================================================
print("\n" + "=" * 60)
print("Model Evaluation on Validation Set")
print("=" * 60)

# Collect predictions
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification Report
print("\nClassification Report:")
print("-" * 40)
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Logo CNN Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✓ Confusion matrix saved as 'confusion_matrix.png'")

# =====================================================
# 10. PREDICTION FUNCTION
# =====================================================
def predict_logo(image_path, show_image=True):
    """
    Predict the logo class for a given image.
    
    Args:
        image_path: Path to the image file
        show_image: Whether to display the image
    
    Returns:
        Predicted class name and confidence
    """
    from tensorflow.keras.utils import load_img, img_to_array
    
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    
    preds = model.predict(arr, verbose=0)
    pred_idx = np.argmax(preds[0])
    pred_class = class_names[pred_idx]
    confidence = preds[0][pred_idx] * 100
    
    print(f"\nImage: {image_path}")
    print(f"Predicted: {pred_class} ({confidence:.2f}% confidence)")
    
    if show_image:
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"{pred_class} ({confidence:.2f}%)")
        plt.axis("off")
        plt.show()
    
    return pred_class, confidence

# Test predictions on validation images
print("\n" + "=" * 60)
print("Sample Predictions")
print("=" * 60)

for cls in class_names[:3]:  # Test first 3 classes
    cls_path = VAL_DIR / cls
    if cls_path.exists():
        images = list(cls_path.glob("*.png")) + list(cls_path.glob("*.jpg"))
        if images:
            predict_logo(str(images[0]), show_image=False)

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print(f"\nFinal Validation Accuracy: {history2.history['val_accuracy'][-1]*100:.2f}%")
print("Files saved:")
print("  - logo_classifier_final.keras")
print("  - best_logo_model_stage1.keras")
print("  - best_logo_model_finetuned.keras")
print("  - training_history.png")
print("  - confusion_matrix.png")
