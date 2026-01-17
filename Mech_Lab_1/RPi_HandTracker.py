import cv2
import mediapipe as mp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device, LED
Device.pin_factory = LGPIOFactory()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== CONFIGURATION ==============
HEADLESS_MODE = False    # Set to True if running without display (SSH)
FRAME_WIDTH = 640        # Lower resolution for better performance
FRAME_HEIGHT = 480

# GPIO Pin Configuration (BCM numbering)
LED1_PIN = 17  # First LED
LED2_PIN = 27  # Second LED
LED3_PIN = 22  # Third LED
# ===========================================

# Initialize LEDs
led1 = LED(LED1_PIN)
led2 = LED(LED2_PIN)
led3 = LED(LED3_PIN)

# Track previous state to avoid restarting blink
previous_led_state = None

def set_leds(prediction):
    global previous_led_state
    
    # Determine current state
    if prediction <= 3:
        current_state = prediction
    else:
        current_state = "blink"
    
    # Only update LEDs if state changed
    if current_state == previous_led_state:
        return
    
    previous_led_state = current_state
    
    if prediction == 0:
        led1.off()
        led2.off()
        led3.off()
    elif prediction == 1:
        led1.on()
        led2.off()
        led3.off()
    elif prediction == 2:
        led1.on()
        led2.on()
        led3.off()
    elif prediction == 3:
        led1.on()
        led2.on()
        led3.on()
    else:
        # Unrecognized gesture (4-7): blink all LEDs
        led1.blink(on_time=0.2, off_time=0.2)
        led2.blink(on_time=0.2, off_time=0.2)
        led3.blink(on_time=0.2, off_time=0.2)

# Initialize camera with optimized settings for Pi
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("ERROR: Could not open camera!")
    print("Check that the camera is connected and not in use by another application.")
    print("On Pi 5, you may need to run: sudo apt install libcamera-tools")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
cap.set(cv2.CAP_PROP_FPS, 30)
print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,           # Track only 1 hand for better performance
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

# ML Model Definition
class Model(nn.Module):
  def __init__(self, in_features=63, h1=128, h2=64, h3=32, out_features=8, dropout=0.2):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.out = nn.Linear(h3, out_features)

  def forward(self, x):
    x = F.gelu(self.fc1(x))
    x = F.gelu(self.fc2(x))
    x = F.gelu(self.fc3(x))
    x = self.out(x)
    return x

# Load the trained model
model = Model()
model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'gesture_model.pth'), weights_only=True))
model.eval()  # Set to evaluation mode

# Class names for display (update these to match your actual gestures)
class_names = ["Fist", "1 Finger", "2 Fingers", "3 Fingers", "4 Fingers", "5 Fingers", "Ok", "Thumbs Up"]

print("Starting hand tracking... Press 'q' to quit.")

try:
    while True:
        success, img = cap.read()
        
        if not success or img is None:
            print("Failed to grab frame")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # Extract all 21 landmarks (x, y, z) into a single row (63 columns)
                row = []
                for lm in handLms.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                
                # Make prediction on the hand landmarks
                hand_data = torch.FloatTensor([row])  # Shape: (1, 63)
                with torch.no_grad():
                    output = model(hand_data)
                    probabilities = F.softmax(output, dim=1)  # Convert to probabilities
                    confidence, predicted_class = torch.max(probabilities, dim=1)
                    predicted_class = predicted_class.item()
                    confidence_pct = confidence.item() * 100
                
                # Display the prediction and confidence on screen
                cv2.putText(img, f"Gesture: {class_names[predicted_class]}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Confidence: {confidence_pct:.1f}%", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Control LEDs based on prediction
                set_leds(predicted_class)
                
                # Print to console in headless mode
                if HEADLESS_MODE:
                    print(f"Gesture: {class_names[predicted_class]} ({confidence_pct:.1f}%)")
                
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
        cTime = time.time()
        if pTime > 0:
            fps = 1 / (cTime - pTime)
            cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        pTime = cTime

        if not HEADLESS_MODE:
            cv2.imshow("Hand Gesture Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    print("Cleaning up...")
    # Turn off all LEDs
    led1.off()
    led2.off()
    led3.off()
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")