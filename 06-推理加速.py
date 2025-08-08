from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train/weights/best.pt")
    model.export(format="engine", dynamic=True, int8=True, data="datasets/fire/fire.yaml")
