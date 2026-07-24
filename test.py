from ultralytics import YOLO

# Load model
model = YOLO(r"C:\Users\Admin\OneDrive\Desktop\du an rung\runs\classify\train\weights\best.pt")

# Predict
results = model.predict(
    source=r"C:\Users\Admin\OneDrive\Desktop\du an rung\test.jpg"
)

# In kết quả
print(results[0].probs.top1)
print(results[0].probs.top1conf)

# Hiển thị ảnh
results[0].show()