# Mouse Webcam Controller

Control your mouse with hand tracking from the webcam using MediaPipe Tasks.

## Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup (Windows)
1. Create/activate venv (optional but recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install deps:
   ```powershell
   pip install -r requirements.txt
   ```

## Setup (Linux)
1. Create/activate venv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```

## Run
Use the venv interpreter so mediapipe/opencv are found:
```powershell
.\.venv\Scripts\python.exe main.py
```
- On Linux:
   ```bash
   ./.venv/bin/python main.py
   ```
- The app downloads `hand_landmarker.task` on first run if missing.
- Move your index finger to control the cursor.
- Pinch (index to middle fingertip) to click.
- Press ESC to quit.

## Notes
- If the webcam is not detected, set the correct camera index in `cv2.VideoCapture(0)`.
- `pyautogui.FAILSAFE` is disabled in code; move mouse to a corner to exit may not work—use ESC.
- On Linux, pyautogui needs X11/XTest; under Wayland use XWayland or a compositor that allows pointer control.
- Ensure your user can access the camera device (often membership in the `video` group).
