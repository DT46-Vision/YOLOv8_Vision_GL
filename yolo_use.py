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