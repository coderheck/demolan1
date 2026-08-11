from ultralytics import YOLO

# Load model classification
model = YOLO("yolov8n-cls.pt")

# Train AI
model.train(
    # ----- param -----
    data = "forest_dataset",
    epochs = 70,
    imgsz = 224,
    batch = 16,
    device = "cpu",
    seed = 1727,
    deterministic = True,
    dropout = 0.3,
    # patience = 10,
    pretrained = True,
    exist_ok = True, # ghi đè folder weight/overwrite weight folder
    val = True,
    workers = 8,
    lr0 = 0.0001,
    optimizer = "AdamW",
    
    # ----- augmentation ----- (có thể sẽ phải chỉnh/probably could tweak later)
    degrees = 90,
    flipud = 0.5,
    fliplr = 0.5,
    mosaic = 0.25,
    auto_augment = "none",
    erasing = 0,

)