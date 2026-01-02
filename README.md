# Mouse Webcam Controller

Control your mouse with hand tracking from the webcam using MediaPipe Tasks.

## Requirements
- Python 3.10+
- Windows tested

## Setup
1. Create/activate venv (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install deps:
   ```powershell
   pip install opencv-python mediapipe pyautogui numpy
   ```

## Run
Use the venv interpreter so mediapipe/opencv are found:
```powershell
.\.venv\Scripts\python.exe main.py
```
- The app downloads `hand_landmarker.task` on first run if missing.
- Move your index finger to control the cursor.
- Pinch (index to middle fingertip) to click.
- Press ESC to quit.

## Notes
- If the webcam is not detected, set the correct camera index in `cv2.VideoCapture(0)`.
- `pyautogui.FAILSAFE` is disabled in code; move mouse to a corner to exit may not work—use ESC.
