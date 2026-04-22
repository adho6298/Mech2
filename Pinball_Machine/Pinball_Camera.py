"""
================================================================================
PINBALL FLIPPER BALL DETECTION - HARDWARE TEST INSTRUCTIONS
================================================================================

DESCRIPTION:
    Detects a white ball on a black background using the Pi camera.
    When the ball enters the left or right detection zone, the corresponding
    relay is triggered to actuate the pinball flipper solenoid.

HARDWARE SETUP:
    - Raspberry Pi with PiCamera2 connected
    - Left flipper relay connected to GPIO 27 (BCM)  ← P2_LEFT_RELAY_PIN in main.py
    - Right flipper relay connected to GPIO 10 (BCM)  ← P2_RIGHT_RELAY_PIN in main.py
    - Relays should be wired to flipper solenoids
    - White ball on dark background for best detection

RUNNING THE PROGRAM:
    python3 Pinball_Camera.py              # normal ball-detection mode
    python3 Pinball_Camera.py --calibrate  # interactive calibration mode

CALIBRATION MODE:
    Shows two windows — raw camera feed (for positioning) and the brightness
    mask (for threshold tuning).  No relays are triggered.
    Controls:
        + / =   raise BRIGHTNESS_THRESHOLD by 1
        -       lower BRIGHTNESS_THRESHOLD by 1
        s       print the current threshold (copy into this file and main.py)
        q       quit

DISPLAY VISUALIZATION:
    - Cyan vertical line = zone boundary (center of frame)
    - Green vertical line = ball detected in LEFT zone
    - Red vertical line = ball detected in RIGHT zone
    - White pixels = detected ball (after threshold)

EXITING THE PROGRAM:
    - Press 'q' or 'Q' while the display window is focused to quit cleanly
    - Press Ctrl+C in the terminal to trigger keyboard interrupt
    - Both methods will safely release relays and clean up GPIO

TUNING:
    Adjust the variables in the "TUNABLE PARAMETERS" section below:
    - FRAME_WIDTH/HEIGHT: Camera resolution (smaller = faster)
    - BRIGHTNESS_THRESHOLD: 0-255, higher = stricter white detection
    - LEFT_RELAY_PIN/RIGHT_RELAY_PIN: GPIO pins (BCM numbering)
    - HIT_TIME: Duration to hold relay for solenoid actuation
    - EXTRA_COOLDOWN: Wait time after hit cycle before allowing re-trigger

TROUBLESHOOTING:
    - No detection? Lower BRIGHTNESS_THRESHOLD or improve lighting
    - False triggers? Raise BRIGHTNESS_THRESHOLD or reduce ambient light
    - Flipper not fully swinging? Increase HIT_TIME
    - Flipper spam? Increase EXTRA_COOLDOWN

================================================================================
"""

import sys
import argparse
import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2, Preview

# ==============================================================================
# DISPLAY DETECTION — for headless systemd service support
# ==============================================================================
def has_display():
    """Check if a display server is available."""
    # Check for DISPLAY variable (X11) or QT_QPA_PLATFORM settings
    has_x11 = os.environ.get('DISPLAY') is not None
    # On headless systems, QT_QPA_PLATFORM will be unset or 'offscreen'
    qt_platform = os.environ.get('QT_QPA_PLATFORM', '')
    is_headless = qt_platform in ('', 'offscreen')
    return has_x11 and is_headless == False

DISPLAY_AVAILABLE = has_display()


"""
=== TUNABLE PARAMETERS ===
All values kept in sync with the constants in main.py.
"""
# Camera frame dimensions — must match FRAME_WIDTH / FRAME_HEIGHT in main.py
# IMX219 (Camera Module v2.1) maximum width is 3280 px
FRAME_WIDTH  = 640
FRAME_HEIGHT = 240

# Calibration mode uses a taller frame so you can see the full scene
CAL_HEIGHT = FRAME_HEIGHT

# Brightness threshold for ball detection (0-255)
# Over threshold = white (ball), under = black (background)
# Must match BRIGHTNESS_THRESHOLD in main.py
BRIGHTNESS_THRESHOLD = 180

# GPIO pin numbers for the P2 (Blue) relays — BCM numbering
# Must match P2_LEFT_RELAY_PIN / P2_RIGHT_RELAY_PIN in main.py
LEFT_RELAY_PIN  = 27   # Physical pin 13
RIGHT_RELAY_PIN = 10   # Physical pin 19

# Hit time (seconds) — relay pulse duration; must match CV_HIT_TIME in main.py
HIT_TIME = 0.05

# Extra cooldown after hit cycle; must match CV_EXTRA_COOLDOWN in main.py
EXTRA_COOLDOWN = 0.4

# Calculated total cooldown (do not modify directly)
DETECTION_COOLDOWN = HIT_TIME * 2 + EXTRA_COOLDOWN


# ==============================================================================
# GPIO SETUP (deferred until needed)
# ==============================================================================
# Note: GPIO initialization is moved into run_detection() so calibration mode
# can run without GPIO permissions. Only detection mode requires GPIO access.


# ==============================================================================
# CALIBRATION MODE
# ==============================================================================

def run_calibration():
    """
    Interactive calibration helper — no relays are triggered here.

    Two windows are shown side-by-side:
      'Calibrate — Raw'  : full-height grayscale frame so you can physically
                           position and aim the camera over the table.
      'Calibrate — Mask' : live brightness mask at the current threshold.
                           Green centroid line = ball detected LEFT zone.
                           Red centroid line   = ball detected RIGHT zone.
                           Cyan line           = zone boundary (centre).

    Keyboard controls (click either window first):
        +  /  =     raise BRIGHTNESS_THRESHOLD by 1
        -           lower BRIGHTNESS_THRESHOLD by 1
        s           print the current threshold value to the terminal
        q           quit calibration

    When done, copy the printed threshold into BRIGHTNESS_THRESHOLD in this
    file AND into BRIGHTNESS_THRESHOLD in main.py so both scripts match.
    """
    if not DISPLAY_AVAILABLE:
        print("[ERROR] Calibration mode requires a display (DISPLAY environment variable or X11)")
        print("[ERROR] This script cannot run in headless mode for calibration.")
        return
    
    threshold = BRIGHTNESS_THRESHOLD
    midpoint  = FRAME_WIDTH // 2
    kernel    = np.ones((3, 3), np.uint8)

    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, CAL_HEIGHT), "format": "YUV420"},
        buffer_count=2,
    )
    picam2.configure(cfg)
    picam2.start()

    print("[CALIBRATE] Camera calibration mode — NO relays will fire")
    print(f"  Starting threshold : {threshold}")
    print("  Controls (click a window first):")
    print("    +/=  raise threshold    -  lower threshold")
    print("    s    print threshold    q  quit")

    cv2.namedWindow("Calibrate — Raw",  cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Calibrate — Mask", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Capture Y-plane (grayscale)
            buffer = picam2.capture_buffer("main")
            y_len  = FRAME_WIDTH * CAL_HEIGHT
            frame  = np.frombuffer(buffer, dtype=np.uint8, count=y_len).reshape(
                (CAL_HEIGHT, FRAME_WIDTH)
            )

            # Build mask at current threshold
            mask   = np.where(frame > threshold, 255, 0).astype(np.uint8)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # ---- Raw window: full greyscale + zone line + threshold label ----
            raw_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.line(raw_bgr, (midpoint, 0), (midpoint, CAL_HEIGHT - 1),
                     (255, 255, 0), 1)
            cv2.putText(raw_bgr,
                        f"Threshold: {threshold}  (+/- adjust | s save | q quit)",
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            # ---- Mask window: binary mask + detection centroids ----
            mask_bgr  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            col_sums  = np.sum(mask, axis=0)
            left_sum  = int(col_sums[:midpoint].sum())
            right_sum = int(col_sums[midpoint:].sum())

            cv2.line(mask_bgr, (midpoint, 0), (midpoint, CAL_HEIGHT - 1),
                     (255, 255, 0), 1)

            if left_sum > 0:
                cx = int(np.dot(np.arange(midpoint), col_sums[:midpoint]) / left_sum)
                cv2.line(mask_bgr, (cx, 0), (cx, CAL_HEIGHT - 1), (0, 255, 0), 2)

            if right_sum > 0:
                cx = midpoint + int(
                    np.dot(np.arange(midpoint), col_sums[midpoint:]) / right_sum
                )
                cv2.line(mask_bgr, (cx, 0), (cx, CAL_HEIGHT - 1), (0, 0, 255), 2)

            label = (f"Threshold:{threshold}  "
                     f"L:{left_sum}  R:{right_sum}  "
                     f"({'DETECTED' if left_sum or right_sum else 'none'})")
            cv2.putText(mask_bgr, label,
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            cv2.imshow("Calibrate — Raw",  raw_bgr)
            cv2.imshow("Calibrate — Mask", mask_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('+'), ord('=')):
                threshold = min(255, threshold + 1)
                print(f"[CALIBRATE] Threshold: {threshold}")
            elif key == ord('-'):
                threshold = max(0, threshold - 1)
                print(f"[CALIBRATE] Threshold: {threshold}")
            elif key in (ord('s'), ord('S')):
                print(f"[CALIBRATE] *** Final threshold: {threshold} ***")
                print(f"  → Set BRIGHTNESS_THRESHOLD = {threshold} in:")
                print(f"       Pinball_Camera.py  (line ~65)")
                print(f"       main.py            (BRIGHTNESS_THRESHOLD constant)")

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"[CALIBRATE] Done. Last threshold used: {threshold}")


# ==============================================================================
# DETECTION MODE
# ==============================================================================

def run_detection():
    """Main ball detection loop — drives relays based on ball position."""
    # Initialize GPIO here (only for detection mode)
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LEFT_RELAY_PIN,  GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(RIGHT_RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)

    midpoint = FRAME_WIDTH // 2
    kernel   = np.ones((3, 3), np.uint8)

    # Cooldown trackers (timestamps when cooldown expires)
    left_cooldown_end  = 0.0
    right_cooldown_end = 0.0

    # FPS calculation
    frame_count = 0
    loop_start  = time.time()
    fps         = 0.0

    # Initialize Picamera2
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "YUV420"},
        buffer_count=2,
    )
    picam2.configure(camera_config)
    if DISPLAY_AVAILABLE:
        picam2.start_preview(Preview.QT)
    picam2.start()

    if DISPLAY_AVAILABLE:
        cv2.namedWindow("Ball Tracker", cv2.WINDOW_AUTOSIZE)
    else:
        print("[INFO] Running in headless mode (no display). Ball detection active, visualization disabled.")

    try:
        while True:
            # Capture Y-plane only (grayscale, fastest path)
            buffer = picam2.capture_buffer("main")
            y_len  = FRAME_WIDTH * FRAME_HEIGHT
            frame  = np.frombuffer(buffer, dtype=np.uint8, count=y_len).reshape(
                (FRAME_HEIGHT, FRAME_WIDTH)
            )

            # Fast threshold using numpy
            mask = np.where(frame > BRIGHTNESS_THRESHOLD, 255, 0).astype(np.uint8)

            # Remove noise with fast morphological open
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Column-projection — ultra-fast horizontal position detection
            column_sums = np.sum(mask, axis=0)
            left_sums   = column_sums[:midpoint]
            right_sums  = column_sums[midpoint:]
            left_total  = int(left_sums.sum())
            right_total = int(right_sums.sum())

            # Build display frame
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.line(display, (midpoint, 0), (midpoint, FRAME_HEIGHT - 1),
                     (255, 255, 0), 1)

            current_time = time.time()

            # Check left zone
            if left_total > 0:
                left_centroid = int(
                    np.dot(np.arange(midpoint), left_sums) / left_total
                )
                cv2.line(display,
                         (left_centroid, 0), (left_centroid, FRAME_HEIGHT - 1),
                         (0, 255, 0), 2)
                if current_time >= left_cooldown_end:
                    print("Ball Left")
                    GPIO.output(LEFT_RELAY_PIN, GPIO.HIGH)
                    time.sleep(HIT_TIME)
                    GPIO.output(LEFT_RELAY_PIN, GPIO.LOW)
                    time.sleep(HIT_TIME)
                    left_cooldown_end = time.time() + EXTRA_COOLDOWN

            # Check right zone
            if right_total > 0:
                right_centroid = midpoint + int(
                    np.dot(np.arange(len(right_sums)), right_sums) / right_total
                )
                cv2.line(display,
                         (right_centroid, 0), (right_centroid, FRAME_HEIGHT - 1),
                         (0, 0, 255), 2)
                if current_time >= right_cooldown_end:
                    print("Ball Right")
                    GPIO.output(RIGHT_RELAY_PIN, GPIO.HIGH)
                    time.sleep(HIT_TIME)
                    GPIO.output(RIGHT_RELAY_PIN, GPIO.LOW)
                    time.sleep(HIT_TIME)
                    right_cooldown_end = time.time() + EXTRA_COOLDOWN

            if left_total == 0 and right_total == 0:
                print("Ball Not Detected")

            # FPS tracking
            frame_count += 1
            elapsed = time.time() - loop_start
            if elapsed >= 1.0:
                fps        = frame_count / elapsed
                frame_count = 0
                loop_start  = time.time()

            if frame_count % 2 == 0:
                if DISPLAY_AVAILABLE:
                    cv2.imshow("Ball Tracker", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")

    finally:
        GPIO.output(LEFT_RELAY_PIN,  GPIO.LOW)
        GPIO.output(RIGHT_RELAY_PIN, GPIO.LOW)
        GPIO.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()
        print("[MAIN] Cleanup complete")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pinball Camera — ball detection and relay control"
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Run interactive calibration mode (camera positioning + threshold tuning, "
             "no relays fired)"
    )
    args = parser.parse_args()

    if args.calibrate:
        run_calibration()
    else:
        run_detection()