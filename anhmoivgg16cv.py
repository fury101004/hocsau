import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tkinter import Tk, filedialog

# Ẩn bớt log TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ==== Load model ====
model = tf.keras.models.load_model("vgg16_cpu_model2.h5")

# ==== Nhãn (giống train_gen.class_indices) ====
label_map = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
}

# Kích thước ảnh input đúng với model đã train
img_size = (128, 128)

# ==== Hàm đọc ảnh Unicode-safe ====
def read_image_unicode(path):
    with open(path, "rb") as f:
        data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ==== Mediapipe Face Detection ====
mp_face = mp.solutions.face_detection


def predict_image(image_path):
    img = read_image_unicode(image_path)
    if img is None:
        print("❌ Không đọc được ảnh:", image_path)
        return

    # Chuyển sang RGB cho Mediapipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_img)

        if not results.detections:
            print("❌ Không tìm thấy khuôn mặt trong ảnh:", image_path)
        else:
            h, w, _ = img.shape
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                # Giới hạn tọa độ để tránh vùng cắt ra ngoài ảnh
                x = max(int(bboxC.xmin * w), 0)
                y = max(int(bboxC.ymin * h), 0)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)

                # Giới hạn chiều rộng, chiều cao vùng cắt không vượt quá ảnh
                bw = min(bw, w - x)
                bh = min(bh, h - y)

                if bw <= 0 or bh <= 0:
                    print("⚠️ Vùng khuôn mặt không hợp lệ, bỏ qua.")
                    continue

                # Cắt khuôn mặt
                face = img[y:y+bh, x:x+bw]
                if face.size == 0:
                    print("⚠️ Không cắt được mặt, bỏ qua.")
                    continue

                # Resize về 128x128 (chuẩn với model)
                face_resized = cv2.resize(face, img_size)
                face_resized = face_resized.astype("float32") / 255.0
                face_resized = np.expand_dims(face_resized, axis=0)  # (1,128,128,3)

                # Dự đoán
                pred = model.predict(face_resized, verbose=0)
                label = label_map[np.argmax(pred)]
                confidence = np.max(pred) * 100

                print(f"👉 Ảnh: {image_path} → {label} ({confidence:.2f}%)")

                # Vẽ khung + nhãn
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {confidence:.1f}%", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Kết quả dự đoán - VGG16 + Mediapipe", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Mở hộp thoại chọn ảnh
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh để dự đoán",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.jfif")]
    )
    if file_path:
        predict_image(file_path)
    else:
        print("❌ Bạn chưa chọn ảnh nào.")
