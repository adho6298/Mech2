import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0
is_recording = False
frame_count = 0
output_file = "hand_data.txt"

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Flatten all 21 landmarks (x, y, z) into a single row (63 columns)
            row = []
            for lm in handLms.landmark:
                row.extend([lm.x, lm.y, lm.z])
            
            # Write to file if recording
            if is_recording:
                with open(output_file, 'a') as f:
                    f.write(' '.join([f'{val:.4f}' for val in row]) + '\n')
                frame_count += 1
                print(f'Recording frame {frame_count}')
            else:
                print(' '.join([f'{val:.4f}' for val in row]))
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    # Display recording status on the image
    status_text = f"Recording: {'ON' if is_recording else 'OFF'} | Frames: {frame_count}"
    cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 if is_recording else 0, 0), 2)
    cv2.putText(img, "Press SPACEBAR to toggle | Q to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Image", img)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == 32:  # Spacebar
        is_recording = not is_recording
        if is_recording:
            print(f"\n=== RECORDING STARTED ===\nData will be saved to: {output_file}\n")
        else:
            print(f"\n=== RECORDING STOPPED ===\nTotal frames recorded: {frame_count}\n")