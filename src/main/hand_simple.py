import cv2
import mediapipe as mp
import time
import numpy as np
import ai_edge_litert.interpreter as tflite
from pathlib import Path

LANDMARKER_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# --- MEDIAPIPE CONFIG (LIVE_STREAM) -----------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the results from HandLandmarker
latest_results = None
DEFAULT_LM_COLOR = (0, 255, 0) # GREEN as default
current_color = DEFAULT_LM_COLOR
last_capture_time = 0      # avoid shaking when capture a photo

# 1. Callback function (Execute when the HandLandmarker return the results)
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_results
    latest_results = result

# 2. Configuration options for Landmarker in LIVE_STREAM mode and result callback
# We define the path to the model, number of hands & confidence coefficents here
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2,
    min_hand_detection_confidence=0.3, # Increase/decrease hand detection sensitivity
    min_hand_presence_confidence=0.6   # Increase/decrease hand landmarking sensitivity
)

# LEGACY --- GESTURE LOGIC FUNCTION ---
def get_gesture(landmarks):
    """Analyse the landmarks' coordinates to determine the gesture"""
    # Finger Tips: 4=Thumb_Tip, 8=Index_Tip, 12=Middle_Tip, 16=Ring_Tip, 20=Pinky_Tip
    # Joint Pips: 3=Thumb_PIP, 6=Index_PIP, 10=Middle_PIP, 14=Ring_PIP, 18=Pinky_PIP
    fingers = []
    
    # """Check for Thumb Up - Compare THUMB landmarks(3 & 4) Y-coordinates"""
    # if landmarks[4].y < landmarks[3].y:
    #     fingers.append(1) # Up
    # else:
    #     fingers.append(0) # Down

    # """Check the other fingers - Looking for V-gesture pattern"""
    for tip, pip in zip([4, 8, 12, 16, 20], [2, 6, 10, 14, 18]):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Gesture determining
    # fingers = [thumb, index, middle, ring, pinky]
    if fingers == [0, 1, 1, 0, 0]:
        return "V_SIGN"
    elif fingers == [1, 0, 0, 0, 0]:
        return "THUMB_UP"
    return "NONE"


# 4. --- MAIN PROGRAM -----------------------------------------------------------------------------
window_name = "Embedded CV - Hand Gesture Detection"
cv2.namedWindow(window_name)

with HandLandmarker.create_from_options(options) as landmarker:
    # Open webcam and process frame by frame
    cap = cv2.VideoCapture(0)
    p_time = 0 # Previous time frame, to calculate FPS

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Calculate FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time       

        """
        Must create timestamp in ms for LIVE_STREAM mode
        Convert frame from BGR (OpenCV) to RGB (MediaPipe Tasks with mp.Image object)
        and send to HandLandmarker for Asynchronous processing (continue to capture the next frame while processing)
        """
        timestamp = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)

        # Process and output the results
        if latest_results and latest_results.hand_landmarks:
            for hand_landmarks in latest_results.hand_landmarks:
                # EXECUTE THE GESTURE LOGIC:
                # 1. Recognise the gesture
                gesture = get_gesture(hand_landmarks)

                # 2. Classify the gesture and and coloring the landmarks
                if gesture == "V_SIGN":
                    # Capture the frame (Cooldown 2s)
                    if time.time() - last_capture_time > 2:
                        current_color = (255, 0, 0)  # Turn landmark to BLUE
                        file_name = f"capture_{int(time.time())}.jpg"
                        cv2.imwrite(file_name, frame)
                        print(f"📸 Capture photo: {file_name}")
                        last_capture_time = time.time()
                elif gesture == "THUMB_UP":
                    # Turn landmark to RED
                    current_color = (0, 0, 255) 
                else:
                    # If no special gesture, return to default color
                    current_color = DEFAULT_LM_COLOR

                # Draw the landmarks
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    # The return mandmarks coordinates are normalized (0->1),
                    #  need to multiply with the frame size h,w
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 3, current_color, -1)

        # 2. Visualize the FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or \
           (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
            break

cap.release()
cv2.destroyAllWindows()