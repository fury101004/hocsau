import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== Thông số ====
test_dir = "test"
img_size = (128, 128)
batch_size = 16

# ==== Load model đã train (compile=False để tránh lỗi) ====
model = tf.keras.models.load_model("vgg16_cpu_model2.h5", compile=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print("✅ Đã load model vgg16_cpu_model2.h5")

# ==== Tạo datagen cho test ====
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=False
)

# ==== Evaluate ====
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"\n🎯 Test Accuracy = {acc:.4f}, Test Loss = {loss:.4f}")

# ==== Dự đoán ====
y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# ==== Classification Report ====
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ==== Confusion Matrix ====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - VGG16")
plt.savefig("confusion_matrix_test.png", dpi=300)
plt.show()

print("✅ Ma trận nhầm lẫn đã được lưu vào confusion_matrix_test.png")
