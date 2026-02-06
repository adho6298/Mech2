import cv2
import numpy as np

def nothing(x):
    """Callback function for trackbars"""
    pass

def main():
    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create windows
    cv2.namedWindow('Original')
    cv2.namedWindow('Mask')
    cv2.namedWindow('Result')
    cv2.namedWindow('Controls', cv2.WINDOW_AUTOSIZE)
    
    # Trackbar names and initial values
    trackbar_info = [
        ('H Min', 0, 179),
        ('H Max', 179, 179),
        ('S Min', 0, 255),
        ('S Max', 50, 255),      # Low saturation for silver
        ('V Min', 150, 255),     # High value for shiny silver
        ('V Max', 255, 255),
        ('Blur', 5, 20),
        ('Erode', 2, 10),
        ('Dilate', 2, 10),
    ]
    
    # Create trackbars
    for name, initial, maximum in trackbar_info:
        cv2.createTrackbar(name, 'Controls', initial, maximum, nothing)
    
    print("Silver Ball Detection")
    print("=====================")
    print("Adjust the trackbars to fine-tune detection for your silver ball")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Crop to center half resolution
        height, width = frame.shape[:2]
        new_width = width // 2
        new_height = height // 2
        x_start = (width - new_width) // 2
        y_start = (height - new_height) // 2
        frame = frame[y_start:y_start + new_height, x_start:x_start + new_width]
        
        # Get trackbar values
        h_min = cv2.getTrackbarPos('H Min', 'Controls')
        h_max = cv2.getTrackbarPos('H Max', 'Controls')
        s_min = cv2.getTrackbarPos('S Min', 'Controls')
        s_max = cv2.getTrackbarPos('S Max', 'Controls')
        v_min = cv2.getTrackbarPos('V Min', 'Controls')
        v_max = cv2.getTrackbarPos('V Max', 'Controls')
        blur_val = cv2.getTrackbarPos('Blur', 'Controls')
        erode_val = cv2.getTrackbarPos('Erode', 'Controls')
        dilate_val = cv2.getTrackbarPos('Dilate', 'Controls')
        
        # Ensure blur is odd and at least 1
        blur_val = max(1, blur_val)
        if blur_val % 2 == 0:
            blur_val += 1
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (blur_val, blur_val), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Create mask for silver color
        lower_silver = np.array([h_min, s_min, v_min])
        upper_silver = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_silver, upper_silver)
        
        # Apply morphological operations to clean up the mask
        if erode_val > 0:
            kernel_erode = np.ones((erode_val, erode_val), np.uint8)
            mask = cv2.erode(mask, kernel_erode, iterations=1)
        
        if dilate_val > 0:
            kernel_dilate = np.ones((dilate_val, dilate_val), np.uint8)
            mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create result image
        result = frame.copy()
        
        # Draw contours and find circles
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area to ignore noise
            if area > 500:
                # Get the minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Calculate circularity to check if it's ball-shaped
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # If circularity is close to 1, it's likely a circle/ball
                    if circularity > 0.5:  # Threshold for circularity
                        # Draw circle around the ball
                        cv2.circle(result, center, radius, (0, 255, 0), 2)
                        # Draw center point
                        cv2.circle(result, center, 5, (0, 0, 255), -1)
                        # Display coordinates and info
                        cv2.putText(result, f"Ball: ({center[0]}, {center[1]})", 
                                    (center[0] - 50, center[1] - radius - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(result, f"R: {radius} C: {circularity:.2f}", 
                                    (center[0] - 50, center[1] - radius - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display HSV values on screen
        info_text = f"HSV: [{h_min}-{h_max}], [{s_min}-{s_max}], [{v_min}-{v_max}]"
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frames
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        
        # Create controls image with labels
        controls_img = np.zeros((320, 300, 3), dtype=np.uint8)
        labels = ['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max', 'Blur', 'Erode', 'Dilate']
        values = [h_min, h_max, s_min, s_max, v_min, v_max, blur_val, erode_val, dilate_val]
        
        for i, (label, val) in enumerate(zip(labels, values)):
            y_pos = 30 + i * 32
            cv2.putText(controls_img, f"{label}: {val}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(controls_img, "Press 'q' quit, 's' save", (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow('Controls', controls_img)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('captured_frame.jpg', frame)
            cv2.imwrite('captured_mask.jpg', mask)
            cv2.imwrite('captured_result.jpg', result)
            print("Frames saved!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
