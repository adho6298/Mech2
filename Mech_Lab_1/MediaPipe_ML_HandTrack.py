import cv2
import mediapipe as mp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
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
model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'gesture_model.pth')))
model.eval()  # Set to evaluation mode

# Class names for display (update these to match your actual gestures)
class_names = ["Fist", "1 Finger", "2 Fingers", "3 Fingers", "4 Fingers", "5 Fingers", "Ok", "Thumbs Up"]

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

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
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    