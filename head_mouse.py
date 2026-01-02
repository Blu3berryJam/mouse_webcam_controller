"""Head-controlled mouse using MediaPipe Face Landmarker.

- Moves cursor continuously based on nose displacement from a neutral pose.
- Auto-calibrates neutral pose when you hold your head still at start.
- Optional left-click when mouth opens beyond a threshold.

Controls:
- Hold head steady for ~1 second at start to auto-calibrate neutral.
- Move head left/right/up/down to move cursor; speed scales with displacement.
- Open mouth to left-click (cooldown guarded).
- Press ESC to quit.

Dependencies:
    pip install opencv-python mediapipe pyautogui numpy
"""

from __future__ import annotations

import cv2
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image
from pathlib import Path
import urllib.request

pyautogui.FAILSAFE = False

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

# Landmark indices (MediaPipe face mesh)
NOSE_TIP = 1
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_BROW = 285
RIGHT_BROW = 55

# Motion tuning
DEADZONE = 0.035         # increased deadzone for stability
GAIN = 1.8               # lower gain for slower movement
MAX_SPEED = 20           # pixels per frame cap
SMOOTHING = 0.35         # slightly more smoothing
DRIFT_ADAPT_RATE = 0.01  # how fast neutral recenters when nearly still

# Click tuning
MOUTH_OPEN_THRESHOLD = 0.06  # mouth open to click/hold
CLICK_COOLDOWN_FRAMES = 18   # frames between clicks
BROW_RAISE_THRESHOLD = 0.025 # eyebrow raise gap for right-click
HOLD_TOGGLE_FRAMES = 12      # frames mouth-open to toggle hold


def _ensure_model() -> Path:
	"""Download the face landmark model if missing."""
	if MODEL_PATH.exists():
		return MODEL_PATH
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
	return MODEL_PATH


def head_mouse() -> None:
	"""Move cursor with head orientation (nose displacement)."""

	model_path = _ensure_model()

	base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
	options = vision.FaceLandmarkerOptions(
		base_options=base_options,
		num_faces=1,
		min_face_detection_confidence=0.6,
		min_face_presence_confidence=0.6,
		min_tracking_confidence=0.6,
		output_face_blendshapes=False,
		output_facial_transformation_matrixes=False,
	)
	landmarker = vision.FaceLandmarker.create_from_options(options)

	cap = cv2.VideoCapture(0)
	screen_w, screen_h = pyautogui.size()

	neutral_nose = None
	stable_frames = 0
	required_stable_frames = 15
	prev_vel = np.array([0.0, 0.0])
	click_cooldown = 0
	last_nose = None
	holding = False
	hold_frames = 0

	while cap.isOpened():
		ok, frame = cap.read()
		if not ok:
			break

		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame, (640, 480))
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = image.Image(image_format=image.ImageFormat.SRGB, data=rgb)
		result = landmarker.detect(mp_image)

		h, w = frame.shape[:2]

		if result.face_landmarks:
			face = result.face_landmarks[0]
			nose = face[NOSE_TIP]
			last_nose = (nose.x, nose.y)
			lip_gap = abs(face[UPPER_LIP].y - face[LOWER_LIP].y)
			brow_gap = abs(face[LEFT_BROW].y - face[RIGHT_BROW].y)

			if neutral_nose is None:
				# Auto-calibration: wait for head to be stable
				if stable_frames == 0:
					neutral_nose = (nose.x, nose.y)
					stable_frames = 1
				else:
					dx = abs(nose.x - neutral_nose[0])
					dy = abs(nose.y - neutral_nose[1])
					if dx < 0.005 and dy < 0.005:
						stable_frames += 1
						neutral_nose = (
							(neutral_nose[0] * (stable_frames - 1) + nose.x) / stable_frames,
							(neutral_nose[1] * (stable_frames - 1) + nose.y) / stable_frames,
						)
					else:
						stable_frames = 1
						neutral_nose = (nose.x, nose.y)
			else:
				# Compute displacement from neutral
				off_x = nose.x - neutral_nose[0]
				off_y = nose.y - neutral_nose[1]

				# Drift correction when nearly centered
				if abs(off_x) < DEADZONE * 1.5 and abs(off_y) < DEADZONE * 1.5:
					neutral_nose = (
						neutral_nose[0] * (1 - DRIFT_ADAPT_RATE) + nose.x * DRIFT_ADAPT_RATE,
						neutral_nose[1] * (1 - DRIFT_ADAPT_RATE) + nose.y * DRIFT_ADAPT_RATE,
					)

				# Apply deadzone
				dx = 0.0 if abs(off_x) < DEADZONE else off_x
				dy = 0.0 if abs(off_y) < DEADZONE else off_y

				# Convert to velocity (pixels per frame)
				vx = np.clip(dx * screen_w * GAIN, -MAX_SPEED, MAX_SPEED)
				vy = np.clip(dy * screen_h * GAIN, -MAX_SPEED, MAX_SPEED)

				# Smooth velocity
				vel = np.array([vx, vy])
				vel = prev_vel * SMOOTHING + vel * (1 - SMOOTHING)
				prev_vel = vel

				# Move cursor relative
				pyautogui.moveRel(vel[0], vel[1])

				# Mouth-open for click/hold toggle
				if click_cooldown > 0:
					click_cooldown -= 1
				if lip_gap > MOUTH_OPEN_THRESHOLD:
					hold_frames += 1
				else:
					hold_frames = 0

				# Short open -> click (if not toggling hold)
				if click_cooldown == 0 and 1 <= hold_frames <= HOLD_TOGGLE_FRAMES // 2:
					pyautogui.click()
					click_cooldown = CLICK_COOLDOWN_FRAMES

				# Long open -> toggle hold (mouse down / up)
				if hold_frames == HOLD_TOGGLE_FRAMES:
					if not holding:
						pyautogui.mouseDown()
						holding = True
					else:
						pyautogui.mouseUp()
						holding = False
					click_cooldown = CLICK_COOLDOWN_FRAMES

				# Eyebrow raise -> right click
				if click_cooldown == 0 and brow_gap > BROW_RAISE_THRESHOLD:
					pyautogui.rightClick()
					click_cooldown = CLICK_COOLDOWN_FRAMES

		# Draw status overlays
		if neutral_nose is None:
			cv2.putText(frame, "Hold head still to calibrate", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
			cv2.putText(frame, f"Stable: {stable_frames}/{required_stable_frames}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		else:
			cv2.putText(frame, "Head control active", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
			cv2.putText(frame, "Mouth: short=open=click, long=hold toggle | Brow raise=right-click | R=Recenter | ESC=Quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('r') or key == ord('R'):
			if last_nose is not None:
				neutral_nose = last_nose
				stable_frames = required_stable_frames
		elif key == 27:  # ESC
			break

		cv2.imshow("Head Mouse", frame)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	head_mouse()
