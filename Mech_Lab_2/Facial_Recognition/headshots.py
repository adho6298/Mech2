import cv2
import os

name = 'Dania' #replace with your name

# Create the directory if it doesn't exist
dataset_path = os.path.join("dataset", name)
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cam = cv2.VideoCapture(0)

cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 500, 300)

img_counter = 0
recording = False

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    # Add text overlay showing recording status and frame count
    status_text = "RECORDING" if recording else "NOT RECORDING"
    status_color = (0, 0, 255) if recording else (0, 255, 0)  # Red if recording, Green if not
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f"Frames saved: {img_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("press space to take a photo", frame)
    
    # Save frame continuously if recording
    if recording:
        img_name = "dataset/"+ name +"/image_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed - toggle recording
        recording = not recording
        if recording:
            print("Recording started...")
        else:
            print("Recording stopped.")

cam.release()

cv2.destroyAllWindows()
