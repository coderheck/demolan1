from ultralytics import YOLO
import cv2
import easyocr
import csv
import os
from datetime import datetime

# =========================
# MODEL
# =========================

model = YOLO(
    r"./runs/classify/train/weights/best.pt"
)

# =========================
# OCR
# =========================

reader = easyocr.Reader(['en'], gpu=False)

# =========================
# CAMERA OBS
# =========================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Không mở được OBS Virtual Camera")
    exit()

# =========================
# CSV
# =========================

CSV_FILE = "forest_log.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "time",
            "forest_class",
            "confidence",
            "telemetry"
        ])

# =========================
# VARIABLES
# =========================

frame_count = 0

forest_class = "unknown"
conf = 0

telemetry = "NO DATA"

# =========================
# LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame_count += 1

    # =========================
    # YOLO (mỗi 10 frame/each 10 frames)
    # =========================

    if frame_count % 10 == 0:

        try:

            small = cv2.resize(frame, (224, 224))

            results = model(
                small,
                verbose=False
            )

            probs = results[0].probs

            class_id = probs.top1

            conf = float(
                probs.top1conf
            )

            forest_class = model.names[class_id]

        except Exception as e:

            print("YOLO ERROR:", e)

    # =========================
    # OCR (mỗi 30 frame/each 30 frames)
    # =========================

    if frame_count % 30 == 0:

        try:

            # CHỈNH LẠI THEO DJI FLY
            crop = frame[430:470, 250:600]

            if crop.size != 0:

                gray = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2GRAY
                )

                _, thresh = cv2.threshold(
                    gray,
                    150,
                    255,
                    cv2.THRESH_BINARY
                )

                texts = reader.readtext(
                    thresh,
                    detail=0
                )

                telemetry = " ".join(texts)

        except Exception as e:

            telemetry = "OCR ERROR"

    # =========================
    # SAVE CSV
    # =========================

    if frame_count % 60 == 0:

        with open(
            CSV_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                datetime.now(),
                forest_class,
                round(conf, 3),
                telemetry
            ])

    # =========================
    # HUD
    # =========================

    cv2.rectangle(
        frame,
        (10, 10),
        (1000, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Forest: {forest_class} ({conf:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Telemetry: {telemetry}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Drone Forest AI",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

# =========================
# CLOSE
# =========================

cap.release()
cv2.destroyAllWindows()