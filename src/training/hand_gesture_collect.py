import cv2
import mediapipe as mp
import time
import numpy as np
import csv
import os

# Data file name
DATA_PATH = "hand_gesture_data.csv"

# Global variable to store processing results from AI thread
latest_results = None
current_color = (0, 255, 0) # Default is green color
last_capture_time = 0      # To prevent shaking when capturing images
p_time = 0 # Previous frame time, used to calculate FPS

# I. --- SYSTEM CONFIGURATION ---------------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 1. Define Callback function (Where AI returns results)
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_results
    latest_results = result

# 2. Configure options for Landmarker with LIVE_STREAM mode and callback
# Here, we specify the model path and maximum number of hands
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2,
    min_hand_detection_confidence=0.3, # Tăng/giảm độ nhạy tìm tay
    min_hand_presence_confidence=0.6   # Tăng/giảm độ nhạy giữ điểm mốc
)

# III. --- FUNCTION TO LOG DATA TO CSV -----------------------------------------------------------------------------
def log_to_csv(label, landmarks):
    """Save 63 coordinates and 1 label to CSV file"""
    file_exists = os.path.isfile(DATA_PATH)
    
    with open(DATA_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Create header if file is new
        if not file_exists:
            header = [f"point_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
            header.append("label")
            writer.writerow(header)
        
        # Flatten data: [[x,y,z], [x,y,z]...] -> [x,y,z,x,y,z...]
        row = []
        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])
        row.append(label)
        writer.writerow(row)

# IV. --- MAIN PROGRAM -------------------------------------------------------------------------------------
window_name = "Embedded AI - Hand Gesture"
cv2.namedWindow(window_name)

with HandLandmarker.create_from_options(options) as landmarker:
    # 4. Open webcam and process each frame
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Create timestamp in ms (Required for LIVE_STREAM)
        # Convert from BGR (OpenCV) to RGB (MediaPipe Tasks use mp.Image object)
        # Send image to Landmarker for processing (Asynchronous)
        timestamp = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        # Process and display results
        if latest_results and latest_results.hand_landmarks:
            for hand_landmarks in latest_results.hand_landmarks:

                # Draw landmarks
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    # Coordinates returned are Normalized (0->1),
                    # need to multiply by actual image size
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 3, current_color, -1)

                # # Display predicted gesture name from ML
                # cv2.putText(frame, f"Gesture: {gesture_label}", (10, 80), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('0'): # Assume 0 is 'None'
            if latest_results and latest_results.hand_landmarks:
                log_to_csv(0, latest_results.hand_landmarks[0])
                print("Logged: Label 0")
        elif key == ord('1'): # Assume 1 is 'V-Sign'
            if latest_results and latest_results.hand_landmarks:
                log_to_csv(1, latest_results.hand_landmarks[0])
                print("Logged: Label 1")
        elif key == ord('2'): # Assume 2 is 'Thumb Up'
            if latest_results and latest_results.hand_landmarks:
                log_to_csv(2, latest_results.hand_landmarks[0])
                print("Logged: Label 2")

        cv2.imshow(window_name, frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or \
           (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
            break

cap.release()
cv2.destroyAllWindows()