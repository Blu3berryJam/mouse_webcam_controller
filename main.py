"""Control the mouse with a hand captured from the webcam.

Dependencies:
	pip install opencv-python mediapipe pyautogui numpy
	
	
----------------------------------------------------------------------
WARNING:
IT PROBABLY WORKS FINE, BUT NO PROMISES.
THIS SOFTWARE IS PROVIDED "AS IS" (NO WARRANTY WHATSOEVER).
ESPECIALLY NOT FOR FITNESS FOR A PARTICULAR PURPOSE 
(LIKE WORLD DOMINATION OR MAKING COFFEE).
----------------------------------------------------------------------
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
FINGER_BEND_THRESHOLD = 0.03  # normalized distance; if tip is below PIP, finger is bent
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Hand landmark indices
INDEX_TIP = 8
INDEX_PIP = 6
PINKY_TIP = 20
PINKY_PIP = 18


def _ensure_model() -> Path:
	"""Download the hand landmark model if it is not present."""
	if MODEL_PATH.exists():
		return MODEL_PATH
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
	return MODEL_PATH


def move_mouse_with_hand() -> None:
	"""Track one hand and move the mouse with the hand center."""

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
	
	# Calibration points: (hand_x, hand_y) -> (screen_x, screen_y)
	calibration_points = []

	def calibrate(frame: np.ndarray, result: vision.HandLandmarkerResult, target_screen_pos: tuple[int, int]) -> None:
		"""Add a calibration point based on current hand position."""
		if result.hand_landmarks:
			hand = result.hand_landmarks[0]
			hand_x = sum(lm.x for lm in hand) / len(hand)
			hand_y = sum(lm.y for lm in hand) / len(hand)
			calibration_points.append(((hand_x, hand_y), target_screen_pos))
			print(f"Calibrated point {len(calibration_points)}: hand({hand_x:.2f}, {hand_y:.2f}) -> screen{target_screen_pos}")

	def hand_to_screen(hand_x: float, hand_y: float) -> tuple[float, float]:
		"""Map hand coordinates to screen coordinates using calibration."""
		if len(calibration_points) < 2:
			# Fallback: simple linear mapping if not calibrated
			return np.interp(hand_x, [0, 1], [0, screen_w]), np.interp(hand_y, [0, 1], [0, screen_h])
		
		# Use two calibration points to compute scaling/offset
		(hx1, hy1), (sx1, sy1) = calibration_points[0]
		(hx2, hy2), (sx2, sy2) = calibration_points[1]
		
		# Linear regression for x and y independently
		scale_x = (sx2 - sx1) / (hx2 - hx1) if hx2 != hx1 else 1
		offset_x = sx1 - hx1 * scale_x
		
		scale_y = (sy2 - sy1) / (hy2 - hy1) if hy2 != hy1 else 1
		offset_y = sy1 - hy1 * scale_y
		
		x = max(0, min(screen_w, hand_x * scale_x + offset_x))
		y = max(0, min(screen_h, hand_y * scale_y + offset_y))
		return x, y

	print("=== Auto-Calibration Mode ===")
	print("Place your hand at TOP-LEFT corner of screen, hold still...")
	
	calibrated = False
	calibration_stage = 0  # 0 = TOP-LEFT, 1 = BOTTOM-RIGHT
	stable_frames = 0
	required_stable_frames = 15  # frames to hold steady before capturing
	last_hand_center = None

	while cap.isOpened():
		ok, frame = cap.read()
		if not ok:
			break

		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame, (640, 480))
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = image.Image(image_format=image.ImageFormat.SRGB, data=rgb)
		result = landmarker.detect(mp_image)

		# Draw calibration UI
		h, w = frame.shape[:2]
		if len(calibration_points) == 0:
			# Show TOP-LEFT corner target
			cv2.circle(frame, (20, 20), 30, (0, 255, 0), 2)
			cv2.putText(frame, "Place hand at TOP-LEFT", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
			progress_text = f"Stable: {stable_frames}/{required_stable_frames}"
			cv2.putText(frame, progress_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		elif len(calibration_points) == 1:
			# Show BOTTOM-RIGHT corner target
			cv2.circle(frame, (w - 20, h - 20), 30, (0, 165, 255), 2)
			cv2.putText(frame, "Place hand at BOTTOM-RIGHT", (w - 300, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
			progress_text = f"Stable: {stable_frames}/{required_stable_frames}"
			cv2.putText(frame, progress_text, (w - 280, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
		else:
			# Calibration complete
			cv2.putText(frame, "Calibration Complete!", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
			cv2.putText(frame, "Starting mouse control...", (w // 2 - 180, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		
		# Show instruction bar
		instruction_text = "Hold hand steady at corner | ESC=Quit"
		cv2.putText(frame, instruction_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		# Auto-calibration logic
		if not calibrated and result.hand_landmarks:
			hand = result.hand_landmarks[0]
			hand_x = sum(lm.x for lm in hand) / len(hand)
			hand_y = sum(lm.y for lm in hand) / len(hand)
			
			if len(calibration_points) == 0:
				# Waiting for TOP-LEFT (check if hand is in top-left region)
				if hand_x < 0.2 and hand_y < 0.2:
					if last_hand_center and abs(hand_x - last_hand_center[0]) < 0.05 and abs(hand_y - last_hand_center[1]) < 0.05:
						stable_frames += 1
					else:
						stable_frames = 1
					
					if stable_frames >= required_stable_frames:
						calibrate(frame, result, (0, 0))
						stable_frames = 0
						last_hand_center = None
				else:
					stable_frames = 0
					
			elif len(calibration_points) == 1:
				# Waiting for BOTTOM-RIGHT (check if hand is in bottom-right region)
				if hand_x > 0.8 and hand_y > 0.8:
					if last_hand_center and abs(hand_x - last_hand_center[0]) < 0.05 and abs(hand_y - last_hand_center[1]) < 0.05:
						stable_frames += 1
					else:
						stable_frames = 1
					
					if stable_frames >= required_stable_frames:
						calibrate(frame, result, (screen_w - 1, screen_h - 1))
						calibrated = True
						stable_frames = 0
						last_hand_center = None
				else:
					stable_frames = 0
			
			last_hand_center = (hand_x, hand_y)
		
		key = cv2.waitKey(1) & 0xFF
		if key == 27:  # ESC
			break
		
		cv2.imshow("Hand Mouse", frame)

		if calibrated and result.hand_landmarks:
			hand = result.hand_landmarks[0]

			# Calculate hand center
			hand_x = sum(lm.x for lm in hand) / len(hand)
			hand_y = sum(lm.y for lm in hand) / len(hand)

			# Map using calibration
			x, y = hand_to_screen(hand_x, hand_y)

			if prev_x is None:
				prev_x, prev_y = x, y
			x = prev_x + (x - prev_x) * (1 - SMOOTHING)
			y = prev_y + (y - prev_y) * (1 - SMOOTHING)
			prev_x, prev_y = x, y

			pyautogui.moveTo(x, y)

			# Index finger bent = click
			idx_tip = hand[INDEX_TIP]
			idx_pip = hand[INDEX_PIP]
			if idx_tip.y > idx_pip.y + FINGER_BEND_THRESHOLD:
				pyautogui.click()

			# Pinky finger bent = right-click
			pinky_tip = hand[PINKY_TIP]
			pinky_pip = hand[PINKY_PIP]
			if pinky_tip.y > pinky_pip.y + FINGER_BEND_THRESHOLD:
				pyautogui.rightClick()

			# Simple on-screen landmarks for visual feedback
			for lm in hand:
				cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
				cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
		
		cv2.imshow("Hand Mouse", frame)

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	move_mouse_with_hand()
