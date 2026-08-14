# ===== script dùng scrcpy =====

import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import subprocess
import easyocr
import csv
from datetime import datetime
import os

# ============================================================
# SCRCPY
# ============================================================

SCRCPY_TITLE = "camcamcam_forest-clc"
scrcpy_process = subprocess.Popen([
    "scrcpy",
    # "-d", # auto-select connected via usb
    "--tcpip=+192.168.137.226:37391", # specify ip:port (different each time)
    # "--tcpip", # auto-select port
    # "-e", # auto-select connected via wifi (tcp/ip)
    # "-s mf8xbyuo8tgqtcba", # device serial
    "--video-codec=h264",
    "--max-size=1600",
    "--max-fps=60",
    "--no-audio",
    "--window-borderless",
    "--always-on-top",
    f"--window-title={SCRCPY_TITLE}",
    # "--window-height=1400",
])
# Give scrcpy time to initialize
time.sleep(3)
print("waiting for scrcpy...")

# ============================================================
# FIND SCRCPY WINDOW
# ============================================================

try:
    windows = gw.getWindowsWithTitle(SCRCPY_TITLE)
    if not windows:
        raise IndexError
    
    win = windows[0]
    print("Found scrcpy window:")
    print(f"Position: {win.left}, {win.top}")
    print(f"Size: {win.width} x {win.height}")
    # win.activate()

except IndexError:
    print("Could not find scrcpy window.")
    scrcpy_process.terminate()
    exit()

# ============================================================
# OCR
# ============================================================

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

# ============================================================
# CSV
# ============================================================

CSV_FILE = "forest_log.csv"
if not os.path.exists(CSV_FILE):
    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "time",
            "lat",
            "lon",
            "forest_class",
            "confidence",
            "telemetry"
        ])

# ============================================================
# MODEL
# ============================================================

from ultralytics import YOLO
model = YOLO(r"./runs/classify/train/weights/best.pt")

# ============================================================
# VARIABLES
# ============================================================

frame_count = 0
forest_class = "unknown"
conf = 0.0
telemetry = "NO DATA"

# Fake GPS for now
lat = 10.7623
lon = 106.6821


# ============================================================
# NORMALIZED OCR CROP
# ============================================================
#
# Instead of:
#
#     frame[430:470, 250:600]
#
# use percentages of the frame dimensions.
#
# These values assume the DJI telemetry is roughly in the
# lower portion of the screen.
#
# Adjust these four values if necessary.
#
# x1 = 0.20  -> 20% from left
# x2 = 0.80  -> 80% from left
# y1 = 0.85  -> 85% from top
# y2 = 0.98  -> 98% from top
#
# ============================================================

# OCR_X1 = 0.20
# OCR_X2 = 0.80
# OCR_Y1 = 0.85
# OCR_Y2 = 0.98
OCR_X1 = 1
OCR_X2 = 1
OCR_Y1 = 1
OCR_Y2 = 1

def get_ocr_crop(frame):
    height, width = frame.shape[:2]
    x1 = int(width * OCR_X1)
    x2 = int(width * OCR_X2)
    y1 = int(height * OCR_Y1)
    y2 = int(height * OCR_Y2)
    return frame[y1:y2, x1:x2]

# ============================================================
# SCREEN CAPTURE
# ============================================================

with mss.MSS() as sct:
    while True:
        # ----------------------------------------------------
        # Get current scrcpy window position
        # ----------------------------------------------------

        monitor = {
            "top": win.top,
            "left": win.left,
            "width": win.width,
            "height": win.height
        }

        # ----------------------------------------------------
        # Capture scrcpy
        # ----------------------------------------------------

        frame = np.array(sct.grab(monitor))
        # processed = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ----------------------------------------------------
        # Frame counter
        # ----------------------------------------------------

        frame_count += 1


        # ====================================================
        # YOLO
        # ====================================================
        #
        # Don't run YOLO on every frame.
        # The forest isn't changing at 60 FPS.
        # Run it every 10 frames instead.
        #
        # ====================================================

        if frame_count % 10 == 0:
            try:
                results = model(
                    # processed, 
                    frame,
                    verbose=False
                )
                probs = results[0].probs
                class_id = probs.top1
                conf = float(probs.top1conf)
                forest_class = model.names[class_id]
            except Exception as e:
                print("YOLO ERROR:", e)

        # ====================================================
        # OCR
        # ====================================================

        if frame_count % 30 == 0:
            try:
                crop = get_ocr_crop(frame)
                if crop.size != 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # Increase contrast for OCR
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

                    if texts:
                        telemetry = " ".join(texts)
            except Exception as e:
                print("OCR ERROR:", e)

        # ====================================================
        # SAVE CSV
        # ====================================================

        if frame_count % 60 == 0:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(),
                    lat,
                    lon,
                    forest_class,
                    round(conf, 3),
                    telemetry
                ])

        # ====================================================
        # DRAW OCR CROP RECTANGLE
        # ====================================================
        # This is extremely useful while configuring the
        # telemetry crop.
        # ====================================================

        height, width = frame.shape[:2]
        x1 = int(width * OCR_X1)
        x2 = int(width * OCR_X2)
        y1 = int(height * OCR_Y1)
        y2 = int(height * OCR_Y2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )

        # ====================================================
        # HUD
        # ====================================================

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

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow("Drone Forest AI", frame)

        # ESC
        if cv2.waitKey(1) == 27:
            break


# ============================================================
# CLEANUP
# ============================================================

cv2.destroyAllWindows()

scrcpy_process.terminate()