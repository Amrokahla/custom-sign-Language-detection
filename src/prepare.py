import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

GESTURES = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

data_dir = 'gesture_data'
os.makedirs(data_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press the corresponding key (A-Z) to record a gesture.")

data = []

def get_landmarks(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = get_landmarks(hand_landmarks)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)
    if key == 13:  # Enter key to exit
        break
    elif chr(key & 0xFF).isalpha():
        gesture = chr(key & 0xFF).upper()
        if gesture in GESTURES and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = get_landmarks(hand_landmarks)
                data.append([gesture] + landmarks)
                print(f"Recorded gesture: {gesture}")

# Save data to CSV after the loop ends
if data:
    pd.DataFrame(data).to_csv(os.path.join(data_dir, 'gesture_data.csv'), index=False, header=False)

cap.release()
cv2.destroyAllWindows()