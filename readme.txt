训练的代码，还需整理和优化

文档地址：【金山文档】 YOlO(GPU)部署文档
https://365.kdocs.cn/l/cqrcO9cj2qqr

数据集获取与参数参考
2023年西交利物浦大学动云科技GMaster战队yolo检测
数据集开源地址: 
链接: https://pan.baidu.com/s/1ayRI1MMw40ae4kuFZCXK_Q?pwd=XPGM
提取码: XPGM
环境需求
1. 创建并激活Conda环境
conda create -n ultralytics-env python=3.10
conda activate ultralytics-env

2. 安装GPU版本的PyTorch（核心步骤）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118

3. 验证CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

4. 安装ultralytics
pip install ultralytics

5. numpy降级
pip uninstall numpy
pip install "numpy<2"环境参考
(ultralytics-env) C:\Users\15196>conda list
# packages in environment at C:\Users\15196\.conda\envs\ultralytics-env:
#
# Name                     Version          Build            Channel
bzip2                      1.0.8            h2bbff1b_6
ca-certificates            2025.7.15        haa95532_0
certifi                    2022.12.7        pypi_0           pypi
charset-normalizer         2.1.1            pypi_0           pypi
contourpy                  1.3.2            pypi_0           pypi
cycler                     0.12.1           pypi_0           pypi
expat                      2.7.1            h8ddb27b_0
filelock                   3.13.1           pypi_0           pypi
fonttools                  4.59.2           pypi_0           pypi
idna                       3.4              pypi_0           pypi
jinja2                     3.1.4            pypi_0           pypi
kiwisolver                 1.4.9            pypi_0           pypi
libffi                     3.4.4            hd77b12b_1
markupsafe                 2.1.5            pypi_0           pypi
matplotlib                 3.10.6           pypi_0           pypi
mpmath                     1.3.0            pypi_0           pypi
networkx                   3.3              pypi_0           pypi
numpy                      1.26.4           pypi_0           pypi
opencv-python              4.12.0.88        pypi_0           pypi
openssl                    3.0.17           h35632f6_0
packaging                  25.0             pypi_0           pypi
pillow                     11.0.0           pypi_0           pypi
pip                        25.2             pyhc872135_0
polars                     1.33.1           pypi_0           pypi
psutil                     7.0.0            pypi_0           pypi
pyparsing                  3.2.3            pypi_0           pypi
python                     3.10.18          h981015d_0
python-dateutil            2.9.0.post0      pypi_0           pypi
pyyaml                     6.0.2            pypi_0           pypi
requests                   2.28.1           pypi_0           pypi
scipy                      1.15.3           pypi_0           pypi
setuptools                 78.1.1           py310haa95532_0
six                        1.17.0           pypi_0           pypi
sqlite                     3.50.2           hda9a48d_1
sympy                      1.13.3           pypi_0           pypi
tk                         8.6.15           hf199647_0
torch                      2.0.1+cu118      pypi_0           pypi
torchvision                0.15.2+cu118     pypi_0           pypi
typing-extensions          4.12.2           pypi_0           pypi
tzdata                     2025b            h04d1e81_0
ucrt                       10.0.22621.0     haa95532_0
ultralytics                8.3.197          pypi_0           pypi
ultralytics-thop           2.0.17           pypi_0           pypi
urllib3                    1.26.13          pypi_0           pypi
vc                         14.3             h2df5915_10
vc14_runtime               14.44.35208      h4927774_10
vs2015_runtime             14.44.35208      ha6b5a95_10
wheel                      0.45.1           py310haa95532_0
xz                         5.6.4            h4754444_1
zlib                       1.2.13           h500123d_2
环境检测
这里加了一个环境检测的脚本
Env.py
import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.device_count())  # 应输出 1
print(torch.cuda.get_device_name(0))  # 应输出 NVIDIA GeForce RTX 4060 Laptop GPU
print(torch.version.cuda)  # 应输出 11.8
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
print(y.sum())  # 如果正常运行，说明 CUDA 没问题如果正常，应该会有如下输出
Detection_ALL/Env.py
True
1
NVIDIA GeForce RTX 4060 Laptop GPU
11.8
tensor(-4912.2031, device='cuda:0')
数据集获取与检测
等我们下好图片和数据集后，我们要把图片按照YOLO可以识别的方式分类
一般标准的会把数据集分为训练集 (train)，验证集(val , 全称validation)和测试集(test)。但是为了方便，这里只用了 train 和 val。
比例：常见划分比例为 70/15/15 或 80/10/10（训练/验证/测试）。大数据集可用 98/1/1。(原推荐的比例是 train : val : test = 8 : 1 : 1，我的比例差不多是32419：55，按照大数据集练的 后面还要优化，欸嘿：)

数据文件结构差不多如下
顺带一提，训练集图像不一定要从1开始，也不一定要连续。只要能与标签名字对应就可以。当然，开源中有改名字的脚本。
project_directory/
├── data/
│   ├── train/                  # 训练集数据
│   │   ├── images/            # 训练集图像（例如）
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   ├── labels/            # 训练集标签（例如）
│   │   │   ├── img1.txt
│   │   │   ├── img2.txt
│   │   │   └── ...
│   ├── validatio (用 val 也可以)/            # 验证集数据
│   │   ├── images/
│   │   │   ├── val_img1.jpg
│   │   │   ├── val_img2.jpg
│   │   │   └── ...
│   │   ├── labels/
│   │   │   ├── val_img1.txt
│   │   │   ├── val_img2.txt
│   │   │   └── ...
│   ├── test/                  # 测试集数据
│   │   ├── images/
│   │   │   ├── test_img1.jpg
│   │   │   ├── test_img2.jpg
│   │   │   └── ...
│   │   ├── labels/
│   │   │   ├── test_img1.txt
│   │   │   ├── test_img2.txt
│   │   │   └── ...
│   ├── train.csv              # 训练集元数据（可选）
│   ├── validation.csv         # 验证集元数据（可选）
│   └── test.csv               # 测试集元数据（可选）图片格式要统一，可以用 .jpg 或 .png，但要统一

文件对应性检测
为了防止图片和标注对不上，加了一个检测图片和标注的脚本
checker.py
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
匹配度可视化
为了防止图片和标注对不上，加了一个单个图片与标注可视化
ReadLabels.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 步骤1: 设置文件路径
image_path = 'D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/train/images/1010.jpg'  # 替换为你的图像路径
label_path = 'D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/train/labels/1010.txt'  # 替换为你的YOLO标注文件路径

# 加载图像
image = cv2.imread(image_path)
if image is None:
    print("无法加载图像！请检查路径:", image_path)
    exit()
height, width, _ = image.shape

# 步骤2: 读取YOLO格式标注
def read_yolo_label(label_path, img_width, img_height):
    boxes = []
    if not os.path.exists(label_path):
        print("标注文件不存在！", label_path)
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # 至少需要 class_id, center_x, center_y, width, height
                continue
            class_id = int(parts[0])
            # 归一化坐标转像素坐标
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            box_width = float(parts[3]) * img_width
            box_height = float(parts[4]) * img_height
            # 计算边界框的左上角和右下角
            x1 = int(center_x - box_width / 2)
            y1 = int(center_y - box_height / 2)
            x2 = int(center_x + box_width / 2)
            y2 = int(center_y + box_height / 2)
            boxes.append((class_id, x1, y1, x2, y2))
    return boxes

# 读取标注
boxes = read_yolo_label(label_path, width, height)

# 步骤3: 可视化
img_copy = image.copy()
for box in boxes:
    class_id, x1, y1, x2, y2 = box
    # 绘制边界框 (绿色)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 添加类标签
    label = f'Class {class_id}'
    cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 如果有关键点（例如YOLOv8-pose，TXT文件包含更多坐标）
# 假设每行格式为: class_id center_x center_y width height x1 y1 v1 x2 y2 v2 ...
def read_yolo_keypoints(label_path, img_width, img_height):
    keypoints = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            kps = []
            for i in range(5, len(parts), 3):  # 跳过class_id和bbox，读取关键点
                x = float(parts[i]) * img_width
                y = float(parts[i + 1]) * img_height
                vis = float(parts[i + 2]) if i + 2 < len(parts) else 0  # 可见性
                kps.append((x, y, vis))
            keypoints.append(kps)
    return keypoints

# 读取关键点（如果适用）
keypoints = read_yolo_keypoints(label_path, width, height)

# 绘制关键点 (红色)
for kps in keypoints:
    for kp in kps:
        x, y, vis = kp
        if vis > 0:  # 只绘制可见关键点
            cv2.circle(img_copy, (int(x), int(y)), 5, (0, 0, 255), -1)

# 步骤4: 显示结果
img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title('YOLOv8标注可视化')
plt.axis('off')
plt.show()

# 步骤5: 验证（可选：与YOLOv8推理结果比较）
# 如果你有YOLOv8模型的检测结果，可以加载模型并推理
# 示例：使用ultralytics YOLOv8（需安装：pip install ultralytics）
"""
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # 替换为你的模型权重
results = model(image_path)
# 绘制推理结果（蓝色框）
for det in results[0].boxes:
    x1, y1, x2, y2 = map(int, det.xyxy[0])
    conf = det.conf[0]
    cls = int(det.cls[0])
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_copy, f'Class {cls} ({conf:.2f})', (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title('YOLOv8标注 vs 推理')
plt.axis('off')
plt.show()
"""可以多换几组数据测试一下标注与图片是否匹配
效果如下图

模型训练与超参数调参
我们已经确定了环境，并且检测了图像和标注的匹配程度，现在要准备训练了。训练和参数其实就只要一个文件，不过为了模块化，我把它分为了类型参数和训练参数两个文件
类型参数
date.yaml
train: D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/train
val: D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/val

nc: 14
names: ['B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BS', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RS']
kpt_shape: [4, 2]上面两个是 train 和 val 文件夹的位置
nc是分别类
names 是类别名称
kpt_shape: [4, 2]   
• kpt_shape：表示关键点（keypoints）的形状或格式，通常用于定义每个检测对象的关键点数量和每个关键点的坐标维度。
[4, 2]：
• 第一个维度 [4]：表示每个检测对象有 4 个关键点。
• 第二个维度 [2]：表示每个关键点的坐标由 2 个值组成，通常是 (x, y) 坐标（即二维平面上的横坐标和纵坐标）。
能量机关是五点模型 ( 四个角点加中心识别点)

训练参数
模型选择有两个上面那个是单识别模型，下面那个是有位姿的模型，第一版用的是上面的，识别装甲板会误识别空区域。    
    model = YOLO('yolov8s.pt')
     model = YOLO('yolov8s-pose.pt')  # 注意使用-pose模型
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('yolov8s.pt')
    model = YOLO('yolov8s-pose.pt')  # 注意使用-pose模型
    
    results = model.train(
        data='D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        # 使用修改后的超参数
        #kpt=0.1,  # 关键点损失权重（仅适用于姿态估计模型）
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        box=0.05,
        cls=0.5,
        hsv_h=0.05,
        hsv_s=0.3,
        hsv_v=0.5,
        degrees=0.3,
        translate=0.1,
        scale=0.1,
        mosaic=1.0
    )
---
YOLO训练参数
1. epochs=100
• 定义：训练的轮次（epochs），表示模型在整个训练集上完整迭代的次数。
• 作用：决定训练的持续时间。更多的 epoch 可能提高模型精度，但过高可能导致过拟合。
• 默认值：YOLOv8 默认 100（与你的设置一致）。
• 建议：
    ◦ 对于关键点检测任务，100 个 epoch 通常合理。如果验证集指标（mAP 或关键点精度）在 50-70 epoch 后收敛，可启用早停（early stopping）以节省时间。
    ◦ 使用命令行设置：yolo train ... epochs=100 或在 custom.yaml 中设置 epochs: 100。
---
2. imgsz=640
• 定义：输入图像大小（image size），表示训练时图像的分辨率（默认正方形，宽高均为 640 像素）。
• 作用：
    ◦ 影响模型的输入尺寸，需与数据集中的图像大小匹配。
    ◦ 更大的 imgsz 提高精度（捕捉更多细节），但增加计算成本和显存需求。
• 默认值：YOLOv8 默认 640。
• 建议：
    ◦ 保持 imgsz=640 与 YOLOv7 一致，确保关键点坐标（归一化）和边界框适配。
    ◦ 如果显存不足，可尝试 imgsz=416 或 320，但可能降低关键点检测精度。
    ◦ 确保数据集图像在预处理时缩放到 640x640（如使用 transforms.Resize）。
    ◦ 设置方式：yolo train ... imgsz=640 或 imgsz: 640。
---
3. batch=16
如果显卡好可拉到32，不然可能会终止训练 （ 别问怎么知道的 ）
• 定义：批大小（batch size），表示每次前向传播处理的样本数量。
• 作用：
    ◦ 影响梯度更新的稳定性：较大的 batch size 使梯度更稳定，但需要更多显存。
    ◦ 小的 batch size 可能增加训练时间，但更适合小显存设备。
• 默认值：YOLOv8 默认 16（与你的设置一致）。
• 建议：
    ◦ batch=16 是平衡性能和显存的常见选择，适合单 GPU（如 device=0）。
    ◦ 如果使用多 GPU 或显存充足，可尝试 batch=32 或更高。
    ◦ 如果显存不足（如 CUDA Out of Memory），降低到 batch=8 或 4。
    ◦ 设置方式：yolo train ... batch=16 或 batch: 16。
---
4. device=0
• 定义：指定训练使用的设备（通常是 GPU 的索引号）。
• 作用：
    ◦ device=0 表示使用第一块 GPU（CUDA 设备编号从 0 开始）。
    ◦ 其他选项：device=cpu（用 CPU）、device=0,1（多 GPU）、device=''（自动选择可用设备）。
• 默认值：YOLOv8 自动选择可用 GPU（若有），否则用 CPU。
• 建议：
    ◦ 如果只有一块 GPU，device=0 是正确设置。
    ◦ 检查 GPU 可用性：运行 nvidia-smi 确认 GPU 编号和显存。
    ◦ 多 GPU 训练可加速，但需确保数据集和 batch size 适配。
    ◦ 设置方式：yolo train ... device=0。
---
5. workers=4 
如果过大可能会终止训练
• 定义：数据加载的工作进程数（number of workers），用于并行加载和预处理数据。
• 作用：
    ◦ 控制数据加载的效率，减少 CPU 瓶颈。
    ◦ 更多的 workers 加快数据准备，但增加内存和 CPU 占用。
• 默认值：YOLOv8 默认 8（你的设置 workers=4 低于默认）。
• 建议：
    ◦ workers=4 适合大多数系统（4 核 CPU 或中等配置）。
    ◦ 如果 CPU 核心数多（如 8 核），可尝试 workers=8 或更高。
    ◦ 如果数据加载慢（如 HDD 硬盘），增加 workers 或使用 SSD。
    ◦ 设置方式：yolo train ... workers=4 或 workers: 4。
---
6. amp=True
• 定义：自动混合精度（Automatic Mixed Precision）训练的开关。
• 作用：
    ◦ 启用 AMP 时，模型在 FP16（半精度浮点）下计算，减少显存占用并加速训练，同时保持接近 FP32 的精度。
    ◦ 适合现代 GPU（如 NVIDIA Volta、Turing、Ampere 架构）。
• 默认值：YOLOv8 默认 amp=True。
• 建议：
    ◦ 保持 amp=True，尤其在 device=0（GPU）上，能显著降低显存需求和训练时间。
    ◦ 如果遇到数值不稳定（如损失 NaN），可尝试 amp=False（全 FP32 训练）。
    ◦ 确保 GPU 支持 AMP（NVIDIA GPU 通常支持）。
    ◦ 设置方式：yolo train ... amp=True 或 amp: true。
epochs 是 训练批次
imgsz 是 图像大小
batch 是 每单次训练数 (如果显卡好，可以拉到32，不然容易溢出终止训练。别问怎么知道的)
后续优化参数
学习率与预热
lr0: 0.01  # 初始学习率，保持不变
lrf: 0.1  # 最终学习率，保持不变
momentum: 0.937  # 动量，保持不变
weight_decay: 0.0005  # 权重衰减，保持不变
warmup_epochs: 3.0  # 预热周期，保持不变
warmup_momentum: 0.8  # 预热动量，保持不变
warmup_bias_lr: 0.1  # 预热偏置学习率，保持不变
损失函数权重
box: 7.5  # YOLOv8 默认值较高，建议测试后调整（YOLOv7: 0.05）
cls: 0.5  # 分类损失权重，保持不变
kpt: 0.5  # 关键点损失权重，建议从 0.10 调整到 0.5 并测试
dfl: 1.5  # YOLOv8 新增的 Distribution Focal Loss 权重，保持默认
cls_pw: 1.0  # 分类正样本权重，保持不变
obj: 1.0  # YOLOv8 使用 CIoULoss，obj 权重默认 1.0（YOLOv7: 0.7）
obj_pw: 1.0  # 目标正样本权重，保持不变
训练参数
iou: 0.7  # YOLOv8 使用 iou 替代 iou_t，建议从 0.20 调整到 0.7
anchor_multiple_threshold: 4.0  # 锚框阈值，保持不变（YOLOv8 仍支持）
fl_gamma: 0.0  # Focal Loss gamma，保持不变
数据增强
hsv_h: 0.015  # YOLOv8 默认值较低，建议测试后从 0.05 调整
hsv_s: 0.7  # YOLOv8 默认值较高，建议测试后从 0.3 调整
hsv_v: 0.4  # YOLOv8 默认值，建议测试后从 0.5 调整
degrees: 0.0  # YOLOv8 默认禁用旋转，建议从 0.3 调整
translate: 0.1  # 平移，保持不变
scale: 0.5  # YOLOv8 默认值较高，建议测试后从 0.1 调整
shear: 0.0  # 剪切，保持不变
perspective: 0.0  # 透视变换，保持不变
flipud: 0.0  # 上下翻转，保持不变
fliplr: 0.5  # YOLOv8 默认启用左右翻转，建议从 0.0 调整
mosaic: 1.0  # Mosaic 增强，保持不变
mixup: 0.0  # Mixup 增强，保持不变

模型评估与使用
模型评估
训练完模型。它会生成一个 run 文件夹，里面有你模型本体、模型的测试数据和测试图片

模型本体
在weights文件夹里的best.pt
last.pt是最后一批次的模型，但best.pt 是整体优化后的模型，更有代表性
模型的测试数据

测试图片 （这个能量机关装甲板以击中好像识别还要优化）

模型使用
用于基础测试
from ultralytics import YOLO
import cv2
import numpy as np

# 加载训练好的模型
model = YOLO('runs/detect/train/weights/best.pt')  # 使用训练后的最佳权重

# 打开默认摄像头（通常索引为0）
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("无法打开摄像头")

# 获取摄像头的帧率和分辨率
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 默认30fps如果无法获取
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 运行 YOLOv8 推理
    results = model.predict(source=frame, imgsz=640, conf=0.25, iou=0.45)  # 调整 conf 和 iou 根据需要

    # 获取检测结果
    for result in results:
        boxes = result.boxes  # 边界框
        class_names = result.names  # 类别名称

        # 遍历每个检测到的边界框
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box)  # 边界框坐标
            label = f"{class_names[int(cls)]} {conf:.2f}"  # 类别和置信度

            # 使用 OpenCV 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边界框，线宽 2
            # 绘制标签
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示视频帧
    cv2.imshow('YOLOv8 Webcam Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
