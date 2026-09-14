"""Virtual Piano — play piano notes with your fingertips.

Shows five colored key zones (C1-G1) at the bottom of the camera window.
Each fingertip plays the note of the zone it is in. A note is triggered
when a fingertip *enters* a zone and repeats only when it leaves and
re-enters, so holding a finger down does not retrigger the sound.

Run:  python Piano/piano.py
Keys: Q or ESC to quit.
"""

import os
import cv2
import mediapipe as mp
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOUND_DIR = os.path.join(BASE_DIR, "sounds")

NOTES = ["C1", "D1", "E1", "F1", "G1"]
FINGERTIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
FINGER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

KEY_HEIGHT_RATIO = 0.25   # keys occupy the bottom 25% of the frame
MARKER_RADIUS = 18


def load_sounds(sound_dir):
    """Load one WAV file per note; a note without a file is skipped."""
    pygame.mixer.init()
    pygame.init()
    sounds = {}
    for note in NOTES:
        path = os.path.join(sound_dir, f"{note}.wav")
        if os.path.exists(path):
            sounds[note] = pygame.mixer.Sound(path)
        else:
            print(f"Sound file not found: {path}")
    return sounds


def open_camera(index=0, width=1280, height=720):
    """Open a webcam, preferring MJPG so low-latency previews stay smooth."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam (index {}). Connect a camera and make sure "
            "no other program is using it.".format(index)
        )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    sounds = load_sounds(SOUND_DIR)
    cap = open_camera()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    # One tracking state per fingertip, per hand: True while the fingertip
    # is inside a key zone (so entering the zone only plays the note once).
    finger_in_key = [{tip: False for tip in FINGERTIPS} for _ in range(2)]

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cv2.namedWindow("Virtual Piano", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Virtual Piano", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame lost — closing.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            key_width = w // len(NOTES)
            key_y = h - int(h * KEY_HEIGHT_RATIO)
            for i, note in enumerate(NOTES):
                x1, x2 = i * key_width, (i + 1) * key_width
                cv2.rectangle(frame, (x1, key_y), (x2, h), (200, 200, 200), -1)
                cv2.putText(frame, note, (x1 + 10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    state = finger_in_key[hand_idx % len(finger_in_key)]
                    for i, tip_idx in enumerate(FINGERTIPS):
                        landmark = hand_landmarks.landmark[tip_idx]
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (x, y), MARKER_RADIUS, FINGER_COLORS[i], -1)

                        zone = min(x // key_width, len(NOTES) - 1)
                        inside = y > key_y
                        if inside and not state[tip_idx]:
                            sound = sounds.get(NOTES[zone])
                            if sound:
                                sound.play()
                        state[tip_idx] = inside

            cv2.putText(frame, "Q / ESC to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Virtual Piano", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    main()
