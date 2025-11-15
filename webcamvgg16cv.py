import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ==== Load model ====
model = tf.keras.models.load_model("vgg16_cpu_model2.h5")

# ==== Nhãn ====
label_map = {
    0: "angry", 1: "disgust", 2: "fear",
    3: "happy", 4: "sad", 5: "surprise", 6: "neutral"
}
img_size = (128, 128)

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def predict_face(face_img):
    face_resized = cv2.resize(face_img, img_size)
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)
    pred = model.predict(face_resized, verbose=0)
    return label_map[np.argmax(pred)]

def predict_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không mở được webcam")
        return

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert sang RGB (mediapipe yêu cầu)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Lấy tọa độ khuôn mặt
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                   int(bboxC.width * w), int(bboxC.height * h)

                    # Crop khuôn mặt
                    face_img = frame[y:y+bh, x:x+bw]
                    if face_img.size == 0:
                        continue

                    label = predict_face(face_img)

                    # Vẽ khung + nhãn
                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Webcam Emotion Recognition - VGG16 + Mediapipe", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_webcam()
