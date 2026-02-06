import cv2
import os

name = "Daniel"  # Name of person


# Initialize the webcam
cam = cv2.VideoCapture(0)
# Set camera resolution
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

# Initialize image counter
img_counter = 0

# Initialize recording flag
isRecording = False

# Create output directory path
output_dir = f"FaceData/{name}/"

os.makedirs(output_dir, exist_ok=True)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    ret, frame = cam.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Error: Could not read frame.")
        break   
    
    # Write image to file if recording
    if isRecording:
        img_name = f"{output_dir}{name}_{img_counter}.jpg"
        cv2.imwrite(img_name, frame)
        img_counter += 1
        print(f"Captured {img_name}")
    
    status_text = f"Recording: {'ON' if isRecording else 'OFF'} | Frames: {img_counter}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255 if isRecording else 0, 0), 2)
    cv2.putText(frame, "Press SPACEBAR to toggle | Q to quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == 32:  # Spacebar
        isRecording = not isRecording
        if isRecording:
            print(f"\n=== RECORDING STARTED ===\nImages will be saved to: {output_dir}\n")
        else:
            print(f"\n=== RECORDING STOPPED ===\nTotal images captured: {img_counter}\n")

# When everything is done, release the capture and close windows
cam.release()
cv2.destroyAllWindows()