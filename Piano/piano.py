import cv2
import mediapipe as mp
import pygame
import os

pygame.mixer.init()
pygame.init()

sounds = {}
notes = ['C1', 'D1', 'E1', 'F1', 'G1']
for note in notes:
    sound_path = os.path.join('sounds', f'{note}.wav')
    if os.path.exists(sound_path):
        sounds[note] = pygame.mixer.Sound(sound_path)
    else:
        print(f"File not found: {sound_path}")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

finger_colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255)]

finger_state = [False]*5

cap = cv2.VideoCapture(0)
cv2.namedWindow("Virtual Piano", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Virtual Piano", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    key_height = 150
    key_width = w // len(notes)
    for i, note in enumerate(notes):
        x1 = i*key_width
        x2 = x1 + key_width
        cv2.rectangle(frame, (x1, h-key_height), (x2, h), (200,200,200), -1)
        cv2.putText(frame, note, (x1+10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            finger_indices = [4,8,12,16,20]  # TIPs
            for i, idx in enumerate(finger_indices):
                x, y = int(landmarks[idx].x*w), int(landmarks[idx].y*h)
                cv2.circle(frame, (x, y), 20, finger_colors[i], -1)

                key_x1 = i*key_width
                key_x2 = key_x1 + key_width
                if key_x1 < x < key_x2 and y > h-key_height:
                    if not finger_state[i]:
                        if notes[i] in sounds:
                            sounds[notes[i]].play()
                        finger_state[i] = True
                else:
                    finger_state[i] = False

    cv2.imshow("Virtual Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
