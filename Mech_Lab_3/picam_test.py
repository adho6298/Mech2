from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()

print("Camera started. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    cv2.imshow("PiCam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
