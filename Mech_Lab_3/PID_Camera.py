from picamera2 import Picamera2
import time
import cv2
import numpy as np
import serial
import threading
from queue import Queue


"""
Thread-safe communication between camera and Arduino threads
"""
ball_position_queue = Queue(maxsize=1)  # Only keep the most recent position
shutdown_event = threading.Event()  # Signal to stop threads


"""
Arduino Comms setup
"""
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # Wait for Arduino to initialize
# Clear any garbage data from initial connection
ser.reset_input_buffer()
ser.reset_output_buffer()





"""
Picam setup
"""
# Initialize Picamera2
picam2 = Picamera2()
# Configure camera for low-resolution grayscale capture
camera_config = picam2.create_video_configuration(
    main={"size": (640, 64), "format": "YUV420"},
    buffer_count=2
)
# Apply configuration
picam2.configure(camera_config)
# Start the camera
picam2.start()


"""
PID and Image Processing Setup
"""
# Brightness threshold for ball detection
threshold_value = 127 # Over 127 = white, under 127 = black

# FPS calculation variables
frame_count = 0
start_time = time.time()
fps = 0.0


"""
Arduino Communication Thread
"""
def arduino_communication_thread():
    """
    Separate thread to handle serial communication with Arduino.
    Reads ball positions from queue and sends to Arduino.
    Also reads and displays debug output from Arduino.
    """
    print("[ARDUINO THREAD] Started")
    
    while not shutdown_event.is_set():
        try:
            # Send ball position to Arduino
            if not ball_position_queue.empty():
                ball_x = ball_position_queue.get(timeout=0.01)
                try:
                    ser.write(f"{ball_x}\n".encode('utf-8'))
                    ser.flush()
                except serial.SerialException as e:
                    print(f"[SERIAL ERROR] {e}")
            
            # Read and print debug output from Arduino
            if ser.in_waiting > 0:
                try:
                    arduino_msg = ser.readline().decode('utf-8').strip()
                    if arduino_msg:  # Only print non-empty messages
                        print(f"[ARDUINO] {arduino_msg}")
                except UnicodeDecodeError:
                    pass  # Ignore decode errors from partial reads
                except Exception as e:
                    print(f"[READ ERROR] {e}")
            else:
                time.sleep(0.001)  # Small delay to prevent busy-waiting
                
        except Exception as e:
            print(f"[ARDUINO THREAD ERROR] {e}")
            time.sleep(0.1)
    
    print("[ARDUINO THREAD] Stopped")


"""
Start Arduino Communication Thread
"""
arduino_thread = threading.Thread(target=arduino_communication_thread, daemon=True)
arduino_thread.start()


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
        y_len = 40960  # 640 * 64
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
            
            # Draw red vertical line at ball position
            cv2.line(display, (ball_x, 0), (ball_x, 63), (0, 0, 255), 2)
            
            # Send ball_x to Arduino thread via queue (non-blocking)
            if ball_position_queue.full():
                try:
                    ball_position_queue.get_nowait()  # Remove old position
                except:
                    pass
            ball_position_queue.put(ball_x)
        
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
    # Signal threads to stop
    shutdown_event.set()
    
    # Wait for Arduino thread to finish
    arduino_thread.join(timeout=2.0)
    
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    ser.close()
    print("[MAIN] Cleanup complete")
    