"""Control the mouse with a hand captured from the webcam using RELATIVE movement.

Dependencies:
    pip install opencv-python mediapipe numpy
    
    Wymaga zainstalowanego i uruchomionego ydotool (daemon).
"""

from __future__ import annotations

import cv2
import numpy as np
import subprocess
import os
import time
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image
from pathlib import Path
import urllib.request

# --- KONFIGURACJA ---
# Czułość myszki (wyższa = szybszy ruch, niższa = precyzyjniejszy)
MOUSE_SENSITIVITY = 1.5 

# Wygładzanie ruchu (0.1 = bardzo pływające/opóźnione, 0.9 = bardzo responsywne/drżące)
SMOOTHING = 0.5

# Próg zgięcia palca do kliknięcia
FINGER_BEND_THRESHOLD = 0.04


# Ścieżka do modelu mediapipe
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Indeksy punktów dłoni
INDEX_TIP = 8
INDEX_PIP = 6
PINKY_TIP = 20
PINKY_PIP = 18
MIDDLE_TIP = 12
MIDDLE_PIP = 10
#serdeczny
RING_TIP = 16
RING_PIP = 14
THuMB_TIP = 4
THUMB_IP = 3
THUMB_START = 1
WRIST = 0

# USTAWIENIE GNIAZDA YDOTOOL (Kluczowe dla działania!)
# Domyślna ścieżka to /run/user/<UID>/.ydotool_socket
UID = os.getuid()
SOCKET_PATH = f"/run/user/{UID}/.ydotool_socket"
os.environ["YDOTOOL_SOCKET"] = SOCKET_PATH
# Użyjemy tej zmiennej do ścieżki demona
YDOTOOLD_BIN = "/usr/bin/ydotoold" # Standardowa ścieżka po instalacji z menedżera pakietów

def ensure_daemon_running():
    """Uruchamia ydotoold w tle, jeśli nie jest jeszcze uruchomiony."""
    print("--- Sprawdzanie i uruchamianie demona ydotoold ---")
    
    # 1. Sprawdzenie, czy demon już działa
    try:
        subprocess.check_output(["pgrep", "ydotoold"])
        print("ydotoold już działa. Kontynuuję.")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ydotoold nie znaleziono. Uruchamiam...")

    # 2. Uruchomienie demona
    try:
        # Uruchamiamy bez sudo, w tle. Demon sam zarządza gniazdem.
        subprocess.Popen([YDOTOOLD_BIN], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         preexec_fn=os.setpgrp)
        time.sleep(1) # Daj czas na uruchomienie
        print("Demon ydotoold uruchomiony.")
    except FileNotFoundError:
        print(f"BŁĄD KRYTYCZNY: Nie znaleziono demona {YDOTOOLD_BIN}. Sprawdź ścieżkę.")
        return

def _ensure_model() -> Path:
    """Download the hand landmark model if it is not present."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

def is_thumb_bent(hand_landmarks, handedness):
    """Sprawdza, czy kciuk jest zgięty."""
    thumb_tip = hand_landmarks[THuMB_TIP]
    thumb_ip = hand_landmarks[THUMB_IP]
    wrist = hand_landmarks[WRIST]

    # Prosta heurystyka: jeśli koniuszek kciuka jest bliżej nadgarstka 
    # w osi X niż jego staw, to jest zgięty.
    # Trzeba uwzględnić, która to ręka.
    if handedness == 'Left': # Lewa ręka (w lustrze prawa)
        return thumb_tip.x > thumb_ip.x
    else: # Prawa ręka (w lustrze lewa)
        return thumb_tip.x < thumb_ip.x

def move_mouse_with_hand() -> None:
    """Track hand and move mouse relatively."""
    model_path = _ensure_model()

    # Konfiguracja Mediapipe
    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # Kamera
    cap = cv2.VideoCapture(0)
    
    # Zmienne do obliczania ruchu
    # Screen width/height używamy tylko do skalowania czułości, nie do absolutnej pozycji
    # Przyjmujemy standardowe 1920x1080 jako bazę odniesienia
    VIRTUAL_SCREEN_W = 1920
    VIRTUAL_SCREEN_H = 1200
    
    prev_screen_x = 0
    prev_screen_y = 0
    is_first_frame = True
    is_left_click_held = False  # Stan przytrzymania lewego przycisku
    
    # Ograniczenie błędów (Deadzone) - ignoruj ruchy mniejsze niż X pikseli
    DEADZONE = 2

    print("=== Startowanie Kontrolera (Tryb Wzg`lędny) ===")
    print("Upewnij się, że demon ydotoold jest uruchomiony.")
    print("Naciśnij ESC, aby wyjść.")
    
    # Timery dla cooldownów
    last_click_time = 0
    last_modifier_time = 0
    CLICK_COOLDOWN = 1.0  # sekundy
    MODIFIER_COOLDOWN = 1.0 # sekundy

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        current_time = time.time()

        # Odbicie lustrzane i zmiana kolorów
        frame = cv2.flip(frame, 1)
        # Zmniejszenie rozdzielczości dla wydajności
        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = image.Image(image_format=image.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        
        h, w = frame.shape[:2]

        # --- Logika dla dwóch rąk ---
        mouse_hand = None
        modifier_hand = None

        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            for i, handedness_obj in enumerate(result.handedness):
                hand_label = handedness_obj[0].category_name
                if hand_label == 'Left':  # Lewa ręka do sterowania (w lustrze prawa)
                    mouse_hand = result.hand_landmarks[i]
                    mouse_hand_handedness = hand_label
                elif hand_label == 'Right':  # Prawa ręka jako modyfikator (w lustrze lewa)
                    modifier_hand = result.hand_landmarks[i]
                    modifier_hand_handedness = hand_label

            # --- Logika ręki modyfikującej (przełącznik hold) ---
            if modifier_hand and (current_time - last_modifier_time > MODIFIER_COOLDOWN):
                if is_thumb_bent(modifier_hand, modifier_hand_handedness):
                    cv2.putText(frame, "THUMB BENT", (w - 250, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    mod_idx_tip = modifier_hand[INDEX_TIP]
                    mod_idx_pip = modifier_hand[INDEX_PIP]
                    if mod_idx_tip.y > mod_idx_pip.y + FINGER_BEND_THRESHOLD:
                        is_left_click_held = not is_left_click_held  # Przełącz stan
                        if is_left_click_held:
                            subprocess.run(["ydotool", "key", "272:1"], stdout=subprocess.DEVNULL) # Wciśnij
                            print("Przełącznik: WCIŚNIĘTO")
                        else:
                            subprocess.run(["ydotool", "key", "272:0"], stdout=subprocess.DEVNULL) # Puść
                            print("Przełącznik: PUSZCZONO")
                        last_modifier_time = current_time # Zresetuj cooldown

        # Wizualizacja stanu przytrzymania
        if is_left_click_held:
            cv2.putText(frame, "[HOLD]", (w // 2 - 50, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if mouse_hand:
            if is_thumb_bent(mouse_hand, mouse_hand_handedness):
                cv2.putText(frame, "THUMB BENT", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # --- Ruch myszką (logika pozostaje taka sama, ale używa mouse_hand) ---
                hand = mouse_hand
                
                # 1. Zamiast środka dłoni nadgarstek (punkt 0)
                hand_x = hand[WRIST].x
                hand_y = hand[WRIST].y - 0.1  # Korekta w górę, aby środek dłoni był bardziej centralny

                # 2. Mapuj na wirtualną rozdzielczość (dla łatwiejszej matematyki)
                target_screen_x = np.interp(hand_x, [0, 1], [0, VIRTUAL_SCREEN_W])
                target_screen_y = np.interp(hand_y, [0, 1], [0, VIRTUAL_SCREEN_H])

                if is_first_frame:
                    prev_screen_x = target_screen_x
                    prev_screen_y = target_screen_y
                    is_first_frame = False
                else:
                    # 3. Wygładzanie (Smoothing)
                    curr_screen_x = prev_screen_x + (target_screen_x - prev_screen_x) * (1 - SMOOTHING)
                    curr_screen_y = prev_screen_y + (target_screen_y - prev_screen_y) * (1 - SMOOTHING)

                    # 4. Oblicz Deltę (Różnicę)
                    delta_x = int((curr_screen_x - prev_screen_x) * MOUSE_SENSITIVITY)
                    delta_y = int((curr_screen_y - prev_screen_y) * MOUSE_SENSITIVITY)

                    # 5. Aktualizacja poprzedniej pozycji
                    prev_screen_x = curr_screen_x
                    prev_screen_y = curr_screen_y

                    # 7. Wysłanie komendy do ydotool
                    if (abs(delta_x) > DEADZONE or abs(delta_y) > DEADZONE):
                        try:
                            command = ["ydotool", "mousemove", "-x", str(delta_x), "-y", str(delta_y)]
                            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            print("Błąd podczas poruszania myszą.")

                # 6. Wizualizacja na obrazie z kamery
                vis_x = int(hand_x * w)
                vis_y = int(hand_y * h)
                cv2.circle(frame, (vis_x, vis_y), 5, (255, 0, 0), cv2.FILLED)

                # --- KLIKANIE (dla ręki sterującej) ---
                if current_time - last_click_time > CLICK_COOLDOWN:
                    idx_tip = hand[INDEX_TIP]
                    idx_pip = hand[INDEX_PIP]
                    pinky_tip = hand[PINKY_TIP]
                    pinky_pip = hand[PINKY_PIP]
                    middle_tip = hand[MIDDLE_TIP]
                    middle_pip = hand[MIDDLE_PIP]
                    ring_tip = hand[RING_TIP]
                    ring_pip = hand[RING_PIP]

                    # Lewy Klik (Palec wskazujący)
                    if idx_tip.y > idx_pip.y + FINGER_BEND_THRESHOLD:
                        subprocess.run(["ydotool", "click", "0xC0"], stdout=subprocess.DEVNULL)
                        cv2.putText(frame, "CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        last_click_time = current_time

                    # Prawy Klik (Mały palec)
                    elif pinky_tip.y > pinky_pip.y + FINGER_BEND_THRESHOLD:
                        subprocess.run(["ydotool", "click", "0xC1"], stdout=subprocess.DEVNULL) 
                        cv2.putText(frame, "R-CLICK", (w-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        last_click_time = current_time

                    # Środkowy klik (serdeczny palec)
                    elif ring_tip.y > ring_pip.y + FINGER_BEND_THRESHOLD:
                        subprocess.run(["ydotool", "click", "0xC2"], stdout=subprocess.DEVNULL)
                        cv2.putText(frame, "M-CLICK", (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        last_click_time = current_time

                    # Double Click (środkowy palec)
                    elif middle_tip.y > middle_pip.y + FINGER_BEND_THRESHOLD:
                        subprocess.run(["ydotool", "click", "0xC0"], stdout=subprocess.DEVNULL)
                        time.sleep(0.05)
                        subprocess.run(["ydotool", "click", "0xC0"], stdout=subprocess.DEVNULL)
                        cv2.putText(frame, "D-CLICK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        last_click_time = current_time

            # Rysowanie punktów dłoni
            for lm in mouse_hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
        
        else:
            # Jeśli ręka zniknie, resetujemy flagę, żeby nie było skoku po powrocie
            is_first_frame = True
            # Opcjonalnie: puść przycisk, jeśli ręka sterująca zniknie
            # if is_left_click_held:
            #     subprocess.run(["ydotool", "key", "272:0"], stdout=subprocess.DEVNULL)
            #     is_left_click_held = False
            #     print("Ręka sterująca zniknęła, puszczam przycisk.")

        # Instrukcje na ekranie
        cv2.putText(frame, "ESC - Wyjscie", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Hand Mouse (Relative)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        
    # Upewnij się, że przycisk jest zwolniony po wyjściu z pętli
    if is_left_click_held:
        subprocess.run(["ydotool", "key", "272:0"], stdout=subprocess.DEVNULL)
        print("Wyjście z programu, puszczam przycisk.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ensure_daemon_running()
    
    model_path = _ensure_model()
    move_mouse_with_hand()