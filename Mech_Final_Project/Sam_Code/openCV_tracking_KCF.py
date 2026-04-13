import cv2


print("cv2 version:", getattr(cv2, "__version__", "NO_VERSION_ATTR"))
print("cv2 path:", getattr(cv2, "__file__", "NO_FILE_ATTR"))
print("Has VideoCapture:", hasattr(cv2, "VideoCapture"))

# Initialize the video source
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing index (0->1), or check camera permissions.")

# Read the first frame
ret, frame = cap.read()
if not ret or frame is None:
    cap.release()
    raise RuntimeError("Could not read first frame from webcam.")

# Select the Region of Interest (ROI) for tracking
roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# roi is (x, y, w, h); if user cancels, w or h may be 0
if roi[2] == 0 or roi[3] == 0:
    cap.release()
    raise RuntimeError("No ROI selected (selection canceled).")

# Create the KCF tracker (handles both new and legacy OpenCV APIs)
if hasattr(cv2, "TrackerKCF_create"):
    tracker = cv2.TrackerKCF_create()
elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
    tracker = cv2.legacy.TrackerKCF_create()
else:
    cap.release()
    raise RuntimeError(
        "KCF tracker not available. Install opencv-contrib-python and ensure you're importing the real cv2."
    )

# Initialize tracker
tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        pass
        cv2.putText(frame, "Tracking failure detected", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()