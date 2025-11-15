import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== Thông số ====
test_dir = "test"
img_size = (128, 128)
batch_size = 1   # để dự đoán từng ảnh
output_file = "test_predictions.txt"

# ==== Load model đã train ====
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

# ==== Dự đoán từng ảnh ====
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# ==== Lưu kết quả ra file text ====
with open(output_file, "w", encoding="utf-8") as f:
    for i, file_path in enumerate(test_gen.filepaths):
        true_label = class_labels[y_true[i]]
        pred_label = class_labels[y_pred[i]]
        confidence = np.max(preds[i]) * 100
        line = f"{os.path.basename(file_path)} | True: {true_label} | Pred: {pred_label} ({confidence:.2f}%)\n"
        f.write(line)
        print(line.strip())

print(f"\n✅ Đã lưu toàn bộ dự đoán test set vào {output_file}")
