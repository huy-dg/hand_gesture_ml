import cv2
import mediapipe as mp
import time
import numpy as np
import ai_edge_litert.interpreter as tflite
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "gesture_classifier.tflite"
LANDMARKER_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# --- 1. INITIATE TFLITE INTERPRETER ----------------------------------------------------------------------
interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label list (Matching the order when trained the ML model)
LABELS = ["None", "V-Sign", "Thumb Up"]
DEFAULT_LM_COLOR = (0, 255, 0) # GREEN as  default
current_color = DEFAULT_LM_COLOR

# --- 2. PREPROCESSING LANDMARKS (TO MATCH THE TRAINED MODEL) ---------------------------------------------
"""
The landmarks coordinates after being collect was pre-process before being used to train the ML model
From absolute, to being relative to the wrist coordinate (meaning wrist becomes the Origin)
Therefore, in new model, we also have to convert the capture landmarks to the relative coordinates
"""
def preprocess_landmarks(hand_landmarks):
    # Extract the 63 raw coordinates
    raw_list = []
    for lm in hand_landmarks:
        raw_list.extend([lm.x, lm.y, lm.z])
    
    # Turn to numpy  array of (21, 3) - 21 landmarks points, 3D coordinates
    landmarks = np.array(raw_list).reshape(21, 3)
    
    # IMPORTANT: Set the wrist (Landmark 0) as Origin (0,0,0)
    wrist = landmarks[0]
    relative_landmarks = landmarks - wrist
    
    # Flatten the relatives landmarks into array (1, 63) and force type float32
    input_data = relative_landmarks.flatten().astype(np.float32)
    return np.expand_dims(input_data, axis=0) # adding the batch dimension (1,1,63)

# 3. --- CONFIG MEDIAPIPE (LIVE_STREAM) -----------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to store the results from HandLandmarker
latest_results = None

# 1. Callback function (Execute when the HandLandmarker return the results)
def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_results
    latest_results = result

# 2.  Configuration options for Landmarker in LIVE_STREAM mode and result callback
# We define the path to the model, number of hands & confidence coefficents here
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2,
    min_hand_detection_confidence=0.3, # Increase/decrease hand detection sensitivity
    min_hand_presence_confidence=0.6   # Increase/decrease hand landmarking sensitivity
)

# 4. --- MAIN PROGRAM -----------------------------------------------------------------------------
window_name = "Embedded ML - Hand Gesture Classifier"
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
                # EXECUTE THE ML MODEL:
                # A. Preprocess coordinates
                input_tensor = preprocess_landmarks(hand_landmarks)

                # B. Run the Inference
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                
                # C. Get the results
                output_data = interpreter.get_tensor(output_details[0]['index'])
                prediction_index = np.argmax(output_data)
                confidence = output_data[0][prediction_index]
                
                gesture_name = LABELS[prediction_index]

                # CLASSIFICATION LOGIC:
                if gesture_name == "V-Sign" and confidence > 0.8:
                    current_color = (255, 0, 0)  # Turn landmark to BLUE
                elif gesture_name == "Thumb Up" and confidence > 0.8:
                    current_color = (0, 0, 255) # Turn landmark to RED
                else:
                    current_color = DEFAULT_LM_COLOR # GREEN as default

                # Draw the landmarks
                for landmark in hand_landmarks:
                    h, w, _ = frame.shape
                    # The return mandmarks coordinates are normalized (0->1),
                    #  need to multiply with the frame size h,w
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 3, current_color, -1)

                # Visualize the gesture classification
                cv2.putText(frame, f"{gesture_name} ({int(confidence*100)}%)", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Visualize the FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or \
           (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1):
            break

cap.release()
cv2.destroyAllWindows()