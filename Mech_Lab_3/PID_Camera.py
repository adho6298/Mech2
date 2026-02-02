from picamera2 import Picamera2
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO

"""
Servo Setup and Functions
"""
# Set up GPIO Mode
GPIO.setmode(GPIO.BCM)

# Define the GPIO pin for the servo (GPIO 17, physical pin 11)
SERVO_PIN = 17

# Set up the servo pin as output
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Create PWM instance with 50Hz frequency (standard for servos)
pwm = GPIO.PWM(SERVO_PIN, 50)

# Start PWM with 0% duty cycle
pwm.start(0)

def set_angle(angle):
    """
    Set servo to a specific angle (0-180 degrees)
    Duty cycle calculation: 2% = 0°, 12% = 180°
    """
    duty = 2 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)


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

# PID variables
kp = 0.8
ki = 0.2
kd = 3100
distance_setpoint = 320  # Center of the 640px width



"""
Main Loop
"""
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
    