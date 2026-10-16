import cv2
import easyocr
import threading
import queue
import time
import socket
import csv
import os
import requests
from datetime import datetime
from ultralytics import YOLO

# =========================
# HELPER: GET LOCAL IP
# =========================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


# =========================
# THREADED RTMP FRAME GRABBER
# =========================
class ThreadedRTMPStream:
    """Continuously pulls frames from the RTMP server, dropping stale frames

    to guarantee zero latency accumulation.
    """
    def __init__(self, stream_url):
        # Force FFmpeg low-latency flags for OpenCV
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32"
        )
        self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()


# =========================
# INITIALIZATION
# =========================
LOCAL_IP = get_local_ip()
RTMP_URL = "rtmp://localhost:1935/live/drone"

LOCAL_API_URL = "http://127.0.0.1:8000/api/annotate-latest"
API_KEY = "forest"
DEVICE_ID = "TDM2421-01"

print("=" * 60)
print(f"1.  make sure mediamtx is already running")
print(f"2.  in DJI Fly, set RTMP URL to:")
print(f"        rtmp://{LOCAL_IP}:1935/live/drone")
print(f"3.  run this script")
print(f"4.  W forest gaming")
print("=" * 60)

# Load YOLO Model
model = YOLO(r"./runs/classify/train/weights/best.pt")

# CSV Logging Initialization
CSV_FILE = "forest_log.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "forest_class", "confidence", "telemetry"])

# =========================
# MAIN STREAM LOOP
# =========================
print(f"Connecting to local RTMP stream: {RTMP_URL} ...")
stream = ThreadedRTMPStream(RTMP_URL)

frame_count = 0
forest_class = "waiting..."
conf = 0.0
send_count = 1

while True:
    ret, frame = stream.read()

    if not ret or frame is None:
        # Stream has not started yet or is re-buffering
        time.sleep(0.05)
        continue

    frame_count += 1

    # YOLO Inference (Every 10 frames)
    if frame_count % 10 == 0:
        try:
            # set displayed forest type
            small_frame = cv2.resize(frame, (224, 224))
            results = model(small_frame, verbose=False)
            probs = results[0].probs
            class_id = probs.top1
            conf = float(probs.top1conf)
            forest_class = model.names[class_id]

            # Send the latest forest classification to the local API.
            try:
                # Normalize model labels to the categories accepted by server.py.
                label = str(forest_class).strip().lower()

                aliases = {
                    "dense": "dense",
                    "medium": "medium",
                    "sparse": "sparse",
                    "unknown": "unknown",
                }

                density = aliases.get(label, "unknown")

                response = requests.post(
                    LOCAL_API_URL,
                    params={
                        "density": density,
                        "confidence": conf,
                    },
                    headers={"X-API-Key": API_KEY},
                    timeout=3,
                )

                print(f"uploaded #{send_count}")
                send_count += 1

                # if response.status_code not in (200, 201):
                #     print(f"Classification upload failed ({response.status_code}), got response: ", response.text)

            except requests.RequestException as e:
                print("Classification API error:", e)
        except Exception as e:
            print("YOLO Inference Error:", e)

    # Log to CSV (Every 60 frames)
    if frame_count % 30 == 0:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), forest_class, round(conf, 3)])

    # Overlay HUD
    cv2.rectangle(frame, (10, 10), (900, 120), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Forest Density: {forest_class} ({conf:.2f})",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Drone RTMP Feed", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()