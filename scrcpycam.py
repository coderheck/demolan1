# ===== test scrcpy =====

import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
import subprocess

# 1. Start scrcpy automatically
# [--window-title=camcamcam_forest-clc] gives the opened window a predictable name
# arglist: [--video-codec=h264 --max-size=1600 --max-fps=60 --no-audio -d --windows-borderless --always-on-top]
scrcpy_process = subprocess.Popen([
    "./scrcpy/scrcpy",
    "--video-codec=h264",
    "--max-size=1600",
    "--max-fps=60",
    "--no-audio",
    "-d",
    "--always-on-top",
    "--window-title=camcamcam_forest-clc",
    "--window-borderless",
])
time.sleep(3)  # Wait for the window to open
print("waiting 3secs...")

# 2. Locate the window
try:
    win = gw.getWindowsWithTitle("camcamcam_forest-clc")[0]
    print("found scrcpy window:")
    win.activate()
except IndexError:
    print("Could not find scrcpy window.")
    exit()

# 3. Capture loop with MSS and OpenCV
with mss.mss() as sct:
    while True:
        # Define bounding box based on scrcpy window position
        monitor = {
            "top": win.top,
            "left": win.left,
            "width": win.width,
            "height": win.height
        }
        
        # Grab the screen data
        img = np.array(sct.grab(monitor))
        
        # Convert from BGRA to BGR for OpenCV processing
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # --- DO YOUR OPENCV PROCESSING HERE ---
        # Example: Gray scale conversion
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Display the processed frame
        cv2.imshow(
            "OpenCV Scrcpy Stream", 
            # gray, # hiển thị filter màu xám 
            frame, # hiển thị màu gốc
        )
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
scrcpy_process.terminate()  # Close scrcpy when done