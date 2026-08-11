import cv2
from ultralytics import YOLO

# Load model
model = YOLO(r".\runs\classify\train\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Predict
    results = model(frame)

    # Get class
    probs = results[0].probs
    top1 = probs.top1
    confidence = probs.top1conf.item()

    class_names = model.names
    label = class_names[top1]

    # Draw text
    text = f"{label} {confidence:.2f}"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Forest AI", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 