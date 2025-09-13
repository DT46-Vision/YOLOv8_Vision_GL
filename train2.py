from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    #model = YOLO('yolov8s-pose.pt')  # 注意使用-pose模型
    
    results = model.train(
        data='D:/code_B/RM_YOLO/XJTLU_2023_Detection_ALL/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        amp=True,
        # 使用修改后的超参数（移除了kpt）
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