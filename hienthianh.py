import os
<<<<<<< HEAD
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Đường dẫn dataset
dataset_dir = r"C:\Users\ADMIN\Downloads\FER2013"

splits = ["train", "test"]

data = []

# Duyệt qua train và test
for split in splits:
    split_path = os.path.join(dataset_dir, split)
    
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            data.append({"Class": class_name, "Split": split, "Count": count})

# Chuyển thành DataFrame để dễ vẽ
df = pd.DataFrame(data)

# Vẽ biểu đồ nhóm (train/test cho từng class)
plt.figure(figsize=(10,6))
sns.barplot(data=df, x="Class", y="Count", hue="Split", palette="Set2")

plt.title("Số lượng ảnh trong FER2013 theo từng tập (Train/Test)")
plt.xlabel("Lớp cảm xúc")
plt.ylabel("Số lượng ảnh")
plt.legend(title="Tập dữ liệu")
=======
import random
import cv2
import matplotlib.pyplot as plt

# 📂 Thay đường dẫn thành thư mục chứa UTKFace
DATASET_DIR = r"C:\Users\ADMIN\Downloads\UTKFace (1)\UTKFace"

# Lấy danh sách ảnh
files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]

print(f"📊 Tổng số ảnh: {len(files)}")

# Lấy ngẫu nhiên 9 ảnh
sample_files = random.sample(files, 9)

plt.figure(figsize=(10,10))
for i, file in enumerate(sample_files):
    img_path = os.path.join(DATASET_DIR, file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

    age = file.split("_")[0]  # Lấy tuổi từ tên file

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"Tuổi: {age}")
    plt.axis("off")

plt.tight_layout()
>>>>>>> 17eb31e786c02a169afae8a5a0194d0b5046ce7a
plt.show()
