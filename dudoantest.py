import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== Thông số ====
test_dir = "test"
img_size = (128, 128)

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
    batch_size=1,
    shuffle=False
)

class_labels = list(test_gen.class_indices.keys())

# ==== Chọn ngẫu nhiên 5 ảnh trong test set ====
sample_indices = random.sample(range(len(test_gen.filenames)), 5)

plt.figure(figsize=(15, 6))
for i, idx in enumerate(sample_indices, 1):
    # Lấy ảnh gốc
    img_path = test_gen.filepaths[idx]
    img = tf.keras.utils.load_img(img_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    preds = model.predict(img_array, verbose=0)
    pred_label = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100
    true_label = os.path.basename(os.path.dirname(img_path))

    # Vẽ ảnh
    plt.subplot(1, 5, i)
    plt.imshow(tf.keras.utils.load_img(img_path))
    plt.axis("off")
    plt.title(f"T:{true_label}\nP:{pred_label}\n({confidence:.1f}%)")

plt.suptitle("Dự đoán ngẫu nhiên 5 ảnh từ tập test", fontsize=16)
plt.tight_layout()
plt.show()
