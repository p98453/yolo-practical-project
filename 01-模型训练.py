from ultralytics import YOLO


model = YOLO("yolo11s.pt")

model.train(
    data="datasets/ball/ball.yaml",
    epochs=100,
    batch=32,
    save=True,
    device="0",
)