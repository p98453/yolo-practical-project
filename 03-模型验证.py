from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/best.pt')
    # 验证
    model.val(
        data='datasets/ball/ball.yaml',
        batch=32,
        device='0',
    )