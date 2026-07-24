from ultralytics import YOLO

# Load model classification
model = YOLO("yolov8n-cls.pt")

# Train AI
model.train(
    data="forest_dataset",
    epochs=50,
    imgsz=224,
    batch=16,
    device="cpu"
)