"""Control the mouse with a hand captured from the webcam.

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

# Tunables
SMOOTHING = 0.2          # 0..1, higher = smoother but laggier
CLICK_PINCH_DIST = 0.04  # normalized distance between index/middle tips to trigger click
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def _ensure_model() -> Path:
	"""Download the hand landmark model if it is not present."""
	if MODEL_PATH.exists():
		return MODEL_PATH
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
	return MODEL_PATH


def move_mouse_with_hand() -> None:
	"""Track one hand and move the mouse with the index fingertip."""

	model_path = _ensure_model()

	base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
	options = vision.HandLandmarkerOptions(
		base_options=base_options,
		num_hands=1,
		min_hand_detection_confidence=0.6,
		min_hand_presence_confidence=0.6,
		min_tracking_confidence=0.6,
	)
	landmarker = vision.HandLandmarker.create_from_options(options)

	cap = cv2.VideoCapture(0)
	screen_w, screen_h = pyautogui.size()
	prev_x, prev_y = None, None

	while cap.isOpened():
		ok, frame = cap.read()
		if not ok:
			break

		frame = cv2.flip(frame, 1)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = image.Image(image_format=image.ImageFormat.SRGB, data=rgb)
		result = landmarker.detect(mp_image)

		if result.hand_landmarks:
			hand = result.hand_landmarks[0]
			idx_tip = hand[8]   # index fingertip
			mid_tip = hand[12]  # middle fingertip

			# Map normalized coords (0..1) to screen pixels
			x = np.interp(idx_tip.x, [0, 1], [0, screen_w])
			y = np.interp(idx_tip.y, [0, 1], [0, screen_h])

			if prev_x is None:
				prev_x, prev_y = x, y
			x = prev_x + (x - prev_x) * (1 - SMOOTHING)
			y = prev_y + (y - prev_y) * (1 - SMOOTHING)
			prev_x, prev_y = x, y

			pyautogui.moveTo(x, y)

			# Pinch gesture triggers click
			pinch_dist = np.hypot(idx_tip.x - mid_tip.x, idx_tip.y - mid_tip.y)
			if pinch_dist < CLICK_PINCH_DIST:
				pyautogui.click()

			# Simple on-screen landmarks for visual feedback
			for lm in hand:
				cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
				cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
		cv2.imshow("Hand Mouse", frame)
		if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	move_mouse_with_hand()
