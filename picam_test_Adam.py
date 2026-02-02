from picamera2 import Picamera2
import time
import cv2
import numpy as np

picam2 = Picamera2()
# Minimize data: small resolution, disable extra processing
camera_config = picam2.create_video_configuration(
    main={"size": (640, 64), "format": "YUV420"},
    buffer_count=2  # Reduce buffer overhead
)
picam2.configure(camera_config)
picam2.start()

# Pre-calculate constants
threshold_value = 127

# FPS calculation variables
frame_count = 0
start_time = time.time()
fps = 0.0

while True:
    # Capture buffer (faster than capture_array)
    buffer = picam2.capture_buffer("main")
    
    # Convert buffer to numpy array manually
    # YUV420: Y plane is width * height, followed by U and V planes
    y_len = 640 * 64
    frame = np.frombuffer(buffer, dtype=np.uint8, count=y_len).reshape((64, 640))
    
    # Fast threshold using numpy
    mask = np.where(frame > threshold_value, 255, 0).astype(np.uint8)
    
    # Remove noise with fast morphological open (optional but recommended)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Ultra-fast horizontal position detection: column projection
    column_sums = np.sum(mask, axis=0)  # Sum each column
    total = column_sums.sum()
    
    # Convert mask to BGR for visualization
    display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    if total > 0:  # Ball detected
        # Weighted average for X position (centroid)
        ball_x = int(np.dot(np.arange(640), column_sums) / total)
        print(f"FPS: {fps:.2f} | Ball X: {ball_x}")
        
        # Draw red vertical line at ball position
        cv2.line(display, (ball_x, 0), (ball_x, 63), (0, 0, 255), 2)
    else:
        print(f"FPS: {fps:.2f} | Ball not detected")
    
    # Only show display every few frames to reduce overhead
    if frame_count % 2 == 0:  # Display every 2nd frame
        cv2.imshow('Mask', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    