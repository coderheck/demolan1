from ultralytics import YOLO

# Load model
model = YOLO(r".\runs\classify\train\weights\best.pt")

# Predict
results = model.predict(
    source=r"test.jpg"
)

# In kết quả/Output
print(results[0].probs.top1)
print(results[0].probs.top1conf)

# Hiển thị ảnh/Show predict image with prediction output
results[0].show()