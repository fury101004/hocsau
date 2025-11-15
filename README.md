# 🎭 Facial Emotion Recognition using VGG16 (FER2013)

## 🧠 Giới thiệu

Dự án này xây dựng hệ thống **nhận diện cảm xúc khuôn mặt** bằng mô hình
**VGG16** (pretrained ImageNet) và fine-tune trên bộ dữ liệu
**FER2013**.\
Mô hình phân loại được **7 cảm xúc**:

-   😠 Angry\
-   🤢 Disgust\
-   😨 Fear\
-   😀 Happy\
-   😢 Sad\
-   😲 Surprise\
-   😐 Neutral

Hệ thống hỗ trợ đầy đủ quy trình: **train → evaluate → predict ảnh mới →
realtime webcam**.

✔ Chạy tốt trên **CPU**\
✔ Không cần GPU vẫn đạt **70--74% accuracy**

------------------------------------------------------------------------

## 🎥 Demo

  Webcam realtime            Dự đoán ảnh
  -------------------------- ----------------------------
  ![demo](demo_webcam.gif)   ![demo2](demo_predict.jpg)

------------------------------------------------------------------------

## 📁 Cấu trúc thư mục

    .
    ├── train/                    # Dữ liệu train (7 thư mục class)
    ├── test/                     # Dữ liệu test
    │
    ├── mohinhvggcv.py            # Train mô hình VGG16
    ├── matraan.py                # Đánh giá chi tiết + biểu đồ Precision/Recall/F1
    ├── test.py                   # Evaluate nhanh + Confusion Matrix
    ├── anhtest.py                # Xuất toàn bộ dự đoán test vào file txt
    ├── dudoantest.py             # Hiển thị ngẫu nhiên 5 ảnh test + dự đoán
    ├── anhmoivgg16cv.py          # Dự đoán ảnh tự chọn
    ├── webcamvgg16cv.py          # Realtime webcam (Mediapipe)
    │
    ├── vgg16_cpu_model2.h5       # Mô hình đã train
    └── *.png                     # Biểu đồ sinh ra trong quá trình train/test

------------------------------------------------------------------------

## 🛠 Cài đặt môi trường

``` bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn mediapipe
```

**Đã kiểm thử ổn định trên:**

-   Python 3.8 -- 3.11\
-   TensorFlow 2.13 -- 2.16\
-   Windows 10/11\
-   Ubuntu 20.04+

------------------------------------------------------------------------

## 🚀 Hướng dẫn sử dụng

### 1. Train mô hình

``` bash
python mohinhvggcv.py
```

------------------------------------------------------------------------

### 2. Đánh giá mô hình

``` bash
python test.py
python matraan.py
```

------------------------------------------------------------------------

### 3. Dự đoán ngẫu nhiên 5 ảnh test

``` bash
python dudoantest.py
```

------------------------------------------------------------------------

### 4. Dự đoán ảnh mới

``` bash
python anhmoivgg16cv.py
```

------------------------------------------------------------------------

### 5. Realtime webcam

``` bash
python webcamvgg16cv.py
```

------------------------------------------------------------------------

### 6. Xuất dự đoán test ra file

``` bash
python anhtest.py
```

------------------------------------------------------------------------

## 📊 Kết quả mong đợi (FER2013 test)

Accuracy trung bình: **\~72%**

------------------------------------------------------------------------

## 📈 Gợi ý cải thiện

-   Face alignment bằng MTCNN / Mediapipe\
-   Dùng EfficientNet\
-   Input size 224×224\
-   Train 50--80 epochs\
-   LR scheduler / EarlyStopping

------------------------------------------------------------------------

## 👨‍💻 Tác giả

-   Sinh viên thực hiện đồ án Deep Learning\
-   Code sạch, có comment tiếng Việt

------------------------------------------------------------------------

> *"Cảm xúc không nói dối -- và giờ máy tính cũng hiểu được chúng!"*
