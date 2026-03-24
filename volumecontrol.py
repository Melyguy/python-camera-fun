import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

if not cap.isOpened():
    print("Cannot open camera")
    exit()

min_dist = 20.0
max_dist = 250.0

# Smoothing: keep track of previous brightness to avoid jitter
current_brightness = sbc.get_brightness(display=0)[0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    flipped_frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                flipped_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
            )

            h, w, _ = flipped_frame.shape
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

            cv2.line(flipped_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.circle(flipped_frame, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(flipped_frame, (x2, y2), 8, (0, 0, 255), cv2.FILLED)

            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Map pinch distance → brightness (0–100)
            target_brightness = int(np.interp(dist, [min_dist, max_dist], [0, 100]))

            # Only update if change is significant (reduces flicker)
            if abs(target_brightness - current_brightness) > 2:
                sbc.set_brightness(target_brightness, display=0)
                current_brightness = target_brightness

            cv2.putText(flipped_frame, f"Pinch distance: {dist:.2f}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(flipped_frame, f"Brightness: {current_brightness}%", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.createButton("reset brightness to max", lambda: sbc.set_brightness(100, display=0), None, cv2.QT_PUSH_BUTTON)

    cv2.imshow('Hand Tracking', flipped_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()