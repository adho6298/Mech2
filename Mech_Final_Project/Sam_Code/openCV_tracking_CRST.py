import cv2

# Initialize the video source 
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame = cap.read()
if not ret or frame is None:
    raise RuntimeError("Could not read from webcam")

# Select the Region of Interest (ROI) for tracking
roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

if roi[2] == 0 or roi[3] == 0:
    raise RuntimeError("No ROI selected")

# Create the CSRT tracker (new + legacy compatible)
if hasattr(cv2, "TrackerCSRT_create"):
    tracker = cv2.TrackerCSRT_create()
elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
    tracker = cv2.legacy.TrackerCSRT_create()

# Initialize the tracker with the first frame and ROI
tracker.init(frame, roi)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # Update the tracker
    success, bbox = tracker.update(frame)

    if success:
        # Tracking successful, draw the bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Tracking failed
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()