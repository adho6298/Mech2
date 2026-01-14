import cv2
import mediapipe as mp
import time
from mediapipe.framework.formats import landmark_pb2

# Create aliases for cleaner code
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the latest result
latest_result = None

# Step 1: Define the callback function for Live Stream mode
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Step 2: Configure Hand Landmarker Options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Step 3: Initialize Video Capture and Task
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Step 4: Run detection asynchronously
        # Timestamps must be monotonically increasing (milliseconds)
        frame_timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Step 5: Visualize the latest result
        if latest_result and latest_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(latest_result.hand_landmarks):
                # Print handedness (left/right hand)
                handedness = latest_result.handedness[hand_idx][0].category_name
                print(f"\n--- Hand {hand_idx + 1} ({handedness}) ---")
                
                # Print image landmarks (normalized coordinates)
                print("Image Landmarks:")
                for landmark_idx, landmark in enumerate(hand_landmarks):
                    print(f"  Landmark {landmark_idx}: x={landmark.x:.6f}, y={landmark.y:.6f}, z={landmark.z:.6f}")
                
                # # Print world landmarks (real-world 3D coordinates)
                # if latest_result.world_landmarks
                #     print("World Landmarks:")
                #     for landmark_idx, landmark in enumerate(latest_result.world_landmarks[hand_idx]):
                #         print(f"  Landmark {landmark_idx}: x={landmark.x:.6f}, y={landmark.y:.6f}, z={landmark.z:.6f}")
                
                # Correct way to create the proto list for the new API
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in hand_landmarks
                ])
                
                # Use the drawing utils directly
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hand Landmarker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()