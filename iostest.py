import cv2
import easyocr
import threading
import queue
import time
import socket
import csv
import os
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

print("=" * 60)
print(f"1. In DJI Fly, set RTMP URL to:")
print(f"   rtmp://{LOCAL_IP}:1935/live/drone")
print(f"2. make sure mediamtx is already running")
print(f"3. run this script")
print(f"4. W forest gaming")
print("=" * 60)

# Load YOLO Model
model = YOLO(r"./runs/classify/train/weights/best.pt")

# OCR Setup in Background Thread
ocr_queue = queue.Queue(maxsize=1)
telemetry_data = "NO DATA"

def ocr_worker():
    global telemetry_data
    reader = easyocr.Reader(['en'], gpu=False)
    while True:
        crop = ocr_queue.get()
        if crop is None:
            break
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            texts = reader.readtext(thresh, detail=0)
            telemetry_data = " ".join(texts) if texts else "NO TEXT"
        except Exception:
            telemetry_data = "OCR ERROR"
        ocr_queue.task_done()

threading.Thread(target=ocr_worker, daemon=True).start()

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
            small_frame = cv2.resize(frame, (224, 224))
            results = model(small_frame, verbose=False)
            probs = results[0].probs
            class_id = probs.top1
            conf = float(probs.top1conf)
            forest_class = model.names[class_id]
        except Exception as e:
            print("YOLO Inference Error:", e)

    # Async OCR (Every 30 frames)
    if frame_count % 30 == 0 and ocr_queue.empty():
        crop = frame[430:470, 250:600]
        if crop.size != 0:
            ocr_queue.put(crop.copy())

    # Log to CSV (Every 60 frames)
    if frame_count % 60 == 0:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), forest_class, round(conf, 3), telemetry_data])

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
    cv2.putText(
        frame,
        f"Telemetry: {telemetry_data}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Live Drone RTMP Feed", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()