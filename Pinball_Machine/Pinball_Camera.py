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
    - Left flipper relay connected to GPIO 17 (BCM)
    - Right flipper relay connected to GPIO 27 (BCM)
    - Relays should be wired to flipper solenoids
    - White ball on black/dark background for best detection

RUNNING THE PROGRAM:
    1. Run: python3 Pinball_Camera.py
    2. A window titled 'Ball Tracker' will open showing the camera feed

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

from picamera2 import Picamera2
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO


"""
=== TUNABLE PARAMETERS ===
Adjust these values to tune detection behavior
"""
# Camera frame dimensions (pixels)
FRAME_WIDTH = 1280  
FRAME_HEIGHT = 720

# Brightness threshold for ball detection (0-255)
# Over threshold = white (ball), under threshold = black (background)
BRIGHTNESS_THRESHOLD = 150

# GPIO pin numbers for relay control (BCM numbering)
LEFT_RELAY_PIN = 17
RIGHT_RELAY_PIN = 27

# Hit time (seconds) - how long to hold relay HIGH for full solenoid actuation
# Also used as wait time after release for solenoid to return to rest position
HIT_TIME = 10.00

# Extra cooldown time (seconds) after hit cycle completes
# Total cooldown = HIT_TIME * 2 + EXTRA_COOLDOWN
EXTRA_COOLDOWN = 0.0

# Calculated total cooldown (do not modify directly)
DETECTION_COOLDOWN = HIT_TIME * 2 + EXTRA_COOLDOWN


"""
GPIO Setup
"""
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(LEFT_RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(RIGHT_RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)


"""
Picam setup
"""
# Initialize Picamera2
picam2 = Picamera2()
# Configure camera for low-resolution grayscale capture
camera_config = picam2.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "YUV420"},
    buffer_count=2
)
# Apply configuration
picam2.configure(camera_config)
# Start the camera
picam2.start()


"""
Detection State
"""
# Zone boundary (vertical midpoint)
midpoint = FRAME_WIDTH // 2

# Cooldown trackers (timestamps when cooldown expires)
left_cooldown_end = 0
right_cooldown_end = 0

# FPS calculation variables
frame_count = 0
start_time = time.time()
fps = 0.0


"""
Main Camera Loop
"""
# Create window once with fixed name
cv2.namedWindow('Ball Tracker', cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Capture buffer (faster than capture_array)
        buffer = picam2.capture_buffer("main")
        
        # Convert buffer to numpy array manually
        # YUV420: Y plane is width * height, followed by U and V planes
        y_len = FRAME_WIDTH * FRAME_HEIGHT
        frame = np.frombuffer(buffer, dtype=np.uint8, count=y_len).reshape((FRAME_HEIGHT, FRAME_WIDTH))
        
        # Fast threshold using numpy
        mask = np.where(frame > BRIGHTNESS_THRESHOLD, 255, 0).astype(np.uint8)
        
        # Remove noise with fast morphological open (optional but recommended)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Ultra-fast horizontal position detection: column projection
        column_sums = np.sum(mask, axis=0)  # Sum each column
        
        # Split into left and right zones
        left_sums = column_sums[:midpoint]
        right_sums = column_sums[midpoint:]
        
        left_total = left_sums.sum()
        right_total = right_sums.sum()
        
        # Convert mask to BGR for visualization
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Draw zone divider line (cyan)
        cv2.line(display, (midpoint, 0), (midpoint, FRAME_HEIGHT - 1), (255, 255, 0), 1)
        
        # Get current time for cooldown checks
        current_time = time.time()
        
        # Check left zone
        if left_total > 0:
            # Calculate centroid within left zone
            left_centroid = int(np.dot(np.arange(midpoint), left_sums) / left_total)
            # Draw green marker for left detection
            cv2.line(display, (left_centroid, 0), (left_centroid, FRAME_HEIGHT - 1), (0, 255, 0), 2)
            
            # Trigger if cooldown has expired
            if current_time >= left_cooldown_end:
                print("Ball Left")
                # Actuate left flipper
                GPIO.output(LEFT_RELAY_PIN, GPIO.HIGH)
                time.sleep(HIT_TIME) # Wait for solenoid to actuate
                GPIO.output(LEFT_RELAY_PIN, GPIO.LOW)
                time.sleep(HIT_TIME)  # Wait for solenoid to return
                left_cooldown_end = time.time() + EXTRA_COOLDOWN
        
        # Check right zone
        if right_total > 0:
            # Calculate centroid within right zone (offset by midpoint)
            right_centroid = midpoint + int(np.dot(np.arange(len(right_sums)), right_sums) / right_total)
            # Draw red marker for right detection
            cv2.line(display, (right_centroid, 0), (right_centroid, FRAME_HEIGHT - 1), (0, 0, 255), 2)
            
            # Trigger if cooldown has expired
            if current_time >= right_cooldown_end:
                print("Ball Right")
                # Actuate right flipper
                GPIO.output(RIGHT_RELAY_PIN, GPIO.HIGH)
                time.sleep(HIT_TIME) # Wait for solenoid to actuate
                GPIO.output(RIGHT_RELAY_PIN, GPIO.LOW)
                time.sleep(HIT_TIME)  # Wait for solenoid to return
                right_cooldown_end = time.time() + EXTRA_COOLDOWN
        
        # No ball detected in either zone
        if left_total == 0 and right_total == 0:
            print("Ball Not Detected")
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Show display (every other frame to reduce overhead)
        if frame_count % 2 == 0:
            cv2.imshow('Ball Tracker', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\n[MAIN] Shutting down...")

finally:
    # Clean up
    GPIO.output(LEFT_RELAY_PIN, GPIO.LOW)
    GPIO.output(RIGHT_RELAY_PIN, GPIO.LOW)
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
    print("[MAIN] Cleanup complete")