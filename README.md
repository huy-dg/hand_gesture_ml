# Hand Gesture Recognition Project

This repository demonstrates a full pipeline for real-time hand gesture recognition using MediaPipe and TensorFlow.

The project uses MediaPipe Tasks to extract 21 hand landmarks from webcam frames, then trains a small neural network in TensorFlow to classify gestures based on normalized landmark coordinates.
Training happens in `src/training/training_hand_gesture.ipynb`, where collected landmark data is preprocessed relative to the wrist position, split into training and test sets, and trained with a simple dense model.
After training, the model is exported to TensorFlow Lite (`gesture_classifier.tflite`) for fast inference in live applications.

Key tools and frameworks:

- MediaPipe Tasks: real-time hand landmark detection and landmark stream handling.
- TensorFlow: model building, training, and evaluation.
- TensorFlow Lite: lightweight inference engine for live gesture prediction.
- OpenCV: webcam capture, visualization, and display.
- NumPy / pandas / scikit-learn: data processing and train/test splitting.

Reference:

- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=vi


The ML model is designed for low-latency live performance, making it suitable for desktop webcam demos.
`src/main/hand_ml.py` runs the trained TFLite model and displays predicted gestures with confidence scores, while `src/main/hand_simple.py` uses rule-based landmark heuristics for comparison.

## Features

- `hand_gesture_collect.py`: capture hand landmark coordinates from webcam and log them into `hand_gesture_data.csv`.
- `training_hand_gesture.ipynb`: preprocess landmark data, train a TensorFlow model for gesture classification, and export a TFLite model.
- `hand_simple.py`: rule-based live hand gesture detector using MediaPipe landmarks and heuristic gesture detection.
- `hand_ml.py`: live gesture classifier using a trained TFLite model (`gesture_classifier.tflite`).
- Includes sample model asset `hand_landmarker.task` and reference landmark diagram `hand-landmarks.png`.

## Repository Structure

- `captured_simple/` - saved capture images from the demo.
- `hand-landmarks.png` - visual reference for MediaPipe hand landmarks.
- `requirements.txt` - Python dependency file.
- `src/`
  - `main/` - runtime scripts and models.
    - `gesture_classifier.tflite` - trained TensorFlow Lite gesture classifier.
    - `hand_landmarker.task` - MediaPipe hand landmark model asset.
    - `hand_ml.py` - live ML-based gesture classification demo.
    - `hand_simple.py` - live webcam gesture detection demo using hand landmark heuristics.
  - `training/` - data collection and training scripts.
    - `hand_gesture_collect.py` - data collection script.
    - `hand_gesture_data.csv` - collected landmark dataset with labels.
    - `training_hand_gesture.ipynb` - notebook for preprocessing, training, and exporting the model.

## Supported Gestures

The project recognizes at least these gestures:

- `None` / no recognized gesture
- `V-Sign`
- `Thumb Up`

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The project uses these packages:

- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- TensorFlow (`tensorflow`)
- TensorFlow Lite runtime / `ai-edge-litert`
- pandas
- scikit-learn

## Setup

1. Open a terminal in the project directory:

   ```bash
   cd /path/to/CV_basic
   ```

2. Activate the provided virtual environment (optional):

   ```bash
   source cvhand/bin/activate
   ```

3. Install dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

> If you already have a working local Python environment, make sure it can access the webcam and supports MediaPipe.

## Usage

### 1. Collect gesture data

Run the data collection script to record landmarks and labels:

```bash
python3 src/training/hand_gesture_collect.py
```

Keyboard controls while collecting data:

- `0` - log label `0` (None)
- `1` - log label `1` (V-Sign)
- `2` - log label `2` (Thumb Up)
- `q` - exit

The script writes rows to `src/training/hand_gesture_data.csv` in the format:

`point_0_x, point_0_y, ..., point_20_z, label`

### 2. Train the model

Open and run the notebook with Jupyter Lab/Notebook or your preferred editor:

```bash
jupyter notebook src/training/training_hand_gesture.ipynb
```

The notebook performs:

- CSV loading
- landmark normalization relative to wrist position
- train/test split
- neural network training
- TFLite export

### 3. Run the heuristic gesture demo

```bash
python3 src/main/hand_simple.py
```

This demo uses heuristic rules on hand landmarks to detect the V sign and thumb-up gestures. It also visualizes landmarks and FPS in a live window.

### 4. Run the ML-based gesture demo

```bash
python3 src/main/hand_ml.py
```

This demo uses the exported `gesture_classifier.tflite` to classify gestures from live webcam landmarks. The recognized gesture and confidence are shown on screen.

## Notes

- `src/main/hand_ml.py` expects `gesture_classifier.tflite` in the same folder (`src/main/`).
- `src/training/hand_gesture_collect.py` and `src/main/hand_ml.py` both require `hand_landmarker.task` in the same folder as the script.
- If the webcam does not open, verify that no other app is using it and that the correct device index is available.

## Suggested Improvements

- improve gesture label definitions and add more classes
- support left/right hand selection
- use a faster model or quantized TFLite export for better mobile/edge performance

## License

No license specified. Add a license file if you want to publish this repository publicly.
