# Lie Detection Using Image Processing

Modular video-based behavior analysis pipeline using OpenCV and MediaPipe.

## Project Structure
- `main.py`: Primary entry point.
- `LieDetectionFinalCode.py`: Compatibility entry point.
- `modules/`: Modular pipeline (`app`, `face_analysis`, `pulse_analysis`, `dashboard`, `config`).

## Features
- Blink frequency monitoring (EAR-based)
- Head pose direction estimation
- Hand-to-face distance check
- Eye gaze direction estimation
- BPM estimation using Eulerian video magnification

## Requirements
- Python 3.9+
- `opencv-python`
- `mediapipe`
- `numpy`

Install:
```bash
pip install opencv-python mediapipe numpy
```

## Run
```bash
python main.py
```

## Notes
- Input video path and thresholds are configured in `modules/config.py`.
- Press `Esc` to stop processing.
