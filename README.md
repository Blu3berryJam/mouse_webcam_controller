## This project version uses ydotool 
This version of the project utilizes [ydotool](https://github.com/ReimuNotMoe/ydotool/tree/master) for input handling to ensure compatibility with most Linux distributions, including Wayland.
##Compatibility Notes:
- Windows / Linux (X11): If you are on Windows or prefer not to use external tools, please check the pyautogui-version branch.
- Note: The pyautogui version was tested exclusively on Windows and Linux systems using X11 and may not function correctly on Wayland.

# Mouse Webcam Controller


Control your mouse with hand tracking from the webcam using MediaPipe Tasks.

Whole project is for study and educational purposes.
There is also a head control module, but while the basic functionality is not perfect, this module is currently performing noticeably worse.


This project is released under the MIT License (see LICENSE file).

## WARNING:
IT PROBABLY WORKS FINE, BUT NO PROMISES.
THIS SOFTWARE IS PROVIDED "AS IS" (NO WARRANTY WHATSOEVER).
ESPECIALLY NOT FOR FITNESS FOR A PARTICULAR PURPOSE 
(LIKE WORLD DOMINATION OR MAKING COFFEE).

## Requirements
- Python 3.10+
- [ydotool](https://github.com/ReimuNotMoe/ydotool/tree/master)
- Dependencies listed in `requirements.txt`

## Setup
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
On Linux:
```bash
./.venv/bin/python main.py
```
- The app downloads `hand_landmarker.task` on first run if missing.
- Move your hand to control the cursor.
- Bend right hand fingers for inputs:
-    Index for left click
-    Middle finger for double left click
-    Ring finger for middle mouse button click
-    Pinky for right click
- Bend right hand thumb to cursor stops following your hand
- Bend left hand index finger and then right hand index finger to activate/deactivate holding down the left mouse button
- Press ESC to quit.

- if ydotool is installed, but not running automaticly try to run daemon manually

## Notes
- If the webcam is not detected, set the correct camera index in `cv2.VideoCapture(0)`.
- Ensure your user can access the camera device (often membership in the `video` group).
