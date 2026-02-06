import cv2
import numpy as np
import time
from threading import Thread, Lock

class CameraCapture:
    def __init__(self, src=1):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.frame = None
        self.lock = Lock()
        self.running = True
        
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def detect_ball(hsv, lower, upper, kernel):
    """Full frame detection - slower but finds ball anywhere"""
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area > 300:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            return int(x), int(y), int(radius)
    return None

def detect_ball_bg_sub(fg_mask, kernel):
    """Detect ball using background subtraction mask - fast for moving objects"""
    mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area > 200:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * 3.14159 * area / (perimeter * perimeter)
                if circularity > 0.4:  # Reasonably circular
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    return int(x), int(y), int(radius)
    return None

def track_ball_roi(frame, last_pos, roi_size, lower, upper, kernel):
    """Track in small ROI around last position - very fast"""
    lx, ly, lr = last_pos
    h, w = frame.shape[:2]
    
    # Define ROI bounds
    x1 = max(0, lx - roi_size)
    y1 = max(0, ly - roi_size)
    x2 = min(w, lx + roi_size)
    y2 = min(h, ly + roi_size)
    
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area > 200:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # Convert back to full frame coordinates
            return int(x + x1), int(y + y1), int(radius)
    return None

def main():
    camera = CameraCapture(1)
    time.sleep(0.2)
    
    LOWER_SILVER = np.array([0, 0, 172], dtype=np.uint8)
    UPPER_SILVER = np.array([179, 255, 255], dtype=np.uint8)
    KERNEL = np.ones((3, 3), np.uint8)
    
    # Background subtractor for motion detection
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=40, detectShadows=False)
    
    ROI_SIZE = 60  # Search radius around last known position
    LOST_FRAMES_THRESHOLD = 10  # Full scan after this many lost frames
    
    last_pos = None
    lost_frames = 0
    prev_time = time.time()
    fps = 0
    
    cv2.namedWindow('Ball Detection')
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        
        # Crop to center
        h, w = frame.shape[:2]
        x1, y1 = w // 4, h // 4
        frame = frame[y1:y1 + h // 2, x1:x1 + w // 2]
        
        ball_pos = None
        mode = "---"
        
        # Apply background subtraction (always update the model)
        fg_mask = bg_subtractor.apply(frame)
        
        # Priority 1: ROI tracking (fastest) - if we have recent position
        if last_pos is not None and lost_frames < LOST_FRAMES_THRESHOLD:
            ball_pos = track_ball_roi(frame, last_pos, ROI_SIZE, LOWER_SILVER, UPPER_SILVER, KERNEL)
            if ball_pos:
                mode = "ROI"
        
        # Priority 2: Background subtraction (fast) - detect moving objects
        if ball_pos is None:
            ball_pos = detect_ball_bg_sub(fg_mask, KERNEL)
            if ball_pos:
                mode = "BG"
        
        # Priority 3: Color detection full scan (slowest) - fallback
        if ball_pos is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ball_pos = detect_ball(hsv, LOWER_SILVER, UPPER_SILVER, KERNEL)
            if ball_pos:
                mode = "CLR"
            lost_frames += 1
        else:
            lost_frames = 0
        
        # Draw if found
        if ball_pos:
            x, y, r = ball_pos
            last_pos = ball_pos
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        # FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 / (curr_time - prev_time + 0.0001)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)} [{mode}]", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Ball Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
