import os

image_dir = "D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/train/images"  # 替换为你的图片文件夹路径
label_dir = "D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/train/labels"  # 替换为你的标注文件夹路径

# 获取所有图片文件
images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
labels = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

# 检查缺失的标注文件
for img_file in images:
    label_file = img_file.replace(".jpg", ".txt")
    if not os.path.exists(os.path.join(label_dir, label_file)):
        print(f"Missing label for {img_file}")

# 统计数量
print(f"Total images: {len(images)}")
print(f"Total labels: {len(labels)}")
print(f"Missing labels: {len(images) - len(labels)}")