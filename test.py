# ===== test model bằng ảnh random =====

from ultralytics import YOLO

# Load model
model = YOLO(r".\runs\classify\train\weights\best.pt")

# Predict
results = model.predict(source = r"test.jpg")

# In kết quả/Output
print(results[0].probs.top1) # 0 = dense, 1 = medium, 2 = sparse
print(results[0].probs.top1conf) # confidence

# Hiển thị ảnh cùng confidence của mỗi class/Show predict image with prediction output
results[0].show()