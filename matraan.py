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

# ==== Load model ====
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

class_labels = list(test_gen.class_indices.keys())

# ==== Dự đoán ====
y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
y_true = test_gen.classes

# ==== Classification Report ====
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
print("\n📊 Classification Report:")
for lbl, metrics in report.items():
    print(lbl, metrics)

# ==== Vẽ Confusion Matrix ====
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Test Set")
plt.savefig("confusion_matrix_test.png", dpi=300)
plt.show()

# ==== Vẽ biểu đồ Precision / Recall / F1-score ====
metrics = ["precision", "recall", "f1-score"]
for metric in metrics:
    scores = [report[label][metric] for label in class_labels]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_labels, y=scores, palette="viridis")
    plt.ylim(0, 1)
    plt.title(f"{metric.capitalize()} per Class - Test Set")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Class")
    plt.savefig(f"{metric}_per_class.png", dpi=300)
    plt.show()
