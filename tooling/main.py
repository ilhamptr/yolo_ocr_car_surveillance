from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="./data.yaml",
    epochs=50,
    imgsz=416,
    batch=16,
    device="mps",
    workers=4,
    cache=True,
    patience=15,
)