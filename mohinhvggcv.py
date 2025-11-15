import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


# Paths
train_dir = "train"
test_dir = "test"

# Parameters
img_size = (128, 128)
batch_size = 16
epochs = 15

# Data generators (augmentation nhẹ)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, color_mode="rgb",
    class_mode="categorical", batch_size=batch_size, shuffle=True
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=img_size, color_mode="rgb",
    class_mode="categorical", batch_size=batch_size, shuffle=False
)

num_classes = train_gen.num_classes

# ======================
# Base VGG16
# ======================
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Freeze toàn bộ trước
for layer in base_model.layers:
    layer.trainable = False

# Chỉ fine-tune block cuối
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# Train
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=epochs,
    class_weight=class_weights
)

# ======================
# Vẽ & Lưu biểu đồ Loss + Accuracy
# ======================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)

# Lưu biểu đồ
plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)
plt.show()

print("Biểu đồ quá trình train đã được lưu vào training_curves.png")

# Evaluate
y_pred = np.argmax(model.predict(test_gen), axis=1)
y_true = test_gen.classes
print("Classification Report:\n",
      classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_gen.class_indices.keys(),
            yticklabels=train_gen.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - VGG16 (CPU)")
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print(" Ma trận nhầm lẫn đã được lưu vào confusion_matrix.png")

# Save model
model.save("vgg16_cpu_model2.h5")
print(" Mô hình đã được lưu vào vgg16_cpu_model2.h5")
