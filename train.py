from ultralytics import YOLO

# Load pretrained model 
model = YOLO("yolov8n-cls.pt")

# Train AI
model.train(
    # ----- param -----
    data = "forest_dataset",
    epochs = 100,
    imgsz = 384,
    batch = 9,
    device = [0],
    seed = 1727,
    deterministic = True,
    dropout = 0.3,
    patience = 35,
    pretrained = True,
    exist_ok = True, # ghi đè folder weight/overwrite weight folder
    val = True,
    workers = 0,
    lr0 = 0.0001,
    optimizer = "auto",
    cos_lr = True,
    
    # ----- augmentation ----- (có thể sẽ phải chỉnh/probably could tweak later)
    degrees = 90,
    flipud = 0.5,
    fliplr = 0.5,
    auto_augment = None,
    mosaic = 0,
    erasing = 0,
    hsv_h = 0,
    # hsv_s = 0.5,
    # hsv_v = 0.5,
)