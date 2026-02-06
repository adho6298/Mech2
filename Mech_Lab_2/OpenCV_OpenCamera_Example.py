import cv2

# Initialize the webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:

    ret, frame = cam.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Error: Could not read frame.")
        break   
    
    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)
    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture and close windows
cam.release()
cv2.destroyAllWindows()