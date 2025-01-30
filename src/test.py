# test.py - Handles real-time gesture recognition
import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp

model = tf.keras.models.load_model('gesture_data/model.h5')
le = joblib.load('gesture_data/label_encoder.pkl')
scaler = joblib.load('gesture_data/scaler.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

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
            landmarks = np.array(get_landmarks(hand_landmarks)).reshape(1, -1)
            landmarks = scaler.transform(landmarks)
            pred = model.predict(landmarks)
            gesture = le.inverse_transform([np.argmax(pred)])[0]
            cv2.putText(frame, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
