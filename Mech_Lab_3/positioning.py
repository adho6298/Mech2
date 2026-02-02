from picamera2 import Picamera2
import time
import cv2
import numpy as np

picam2 = Picamera2()
# RGB configuration for full color view
camera_config = picam2.create_video_configuration(
    main={"size": (640, 64), "format": "RGB888"},
    buffer_count=2  # Reduce buffer overhead
)
picam2.configure(camera_config)
picam2.start()

# FPS calculation variables
frame_count = 0
start_time = time.time()
fps = 0.0

while True:
    # Capture full RGB frame
    frame = picam2.capture_array()
    
    # Convert to BGR for display
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    print(f"FPS: {fps:.2f}")
    
    # Display camera view
    cv2.imshow('Camera View', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    