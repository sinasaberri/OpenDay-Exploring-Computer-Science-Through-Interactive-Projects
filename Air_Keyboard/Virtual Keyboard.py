import cv2
import mediapipe as mp
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

persian_map = {
    "Q":"ض","W":"ص","E":"ث","R":"ق","T":"ف","Y":"غ","U":"ع","I":"ه","O":"خ","P":"ح",
    "A":"ش","S":"س","D":"ی","F":"ب","G":"ل","H":"ا","J":"ت","K":"ن","L":"م",
    "Z":"ظ","X":"ط","C":"ز","V":"ر","B":"ذ","N":"د","M":"پ"
}

keyboard_rows = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "<-", "Lang", "Enter"]
]

typed_text = ""
lang = "EN"
cooldown = 0
current_key = None

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
hands = mp_hands.Hands()

def draw_keyboard(image, pressed_key=None):
    positions = {}
    h, w, _ = image.shape
    row_height = h // (len(keyboard_rows) + 2)
    y = row_height * 2

    for row in keyboard_rows:
        key_width = w // len(row)
        x = 0
        for key in row:
            color = (255, 0, 0)
            if pressed_key == key:
                color = (0, 255, 0)

            cv2.rectangle(image, (x, y), (x+key_width, y+row_height), color, 2)

            if key in persian_map and lang == "FA":
                label = f"{key}/{persian_map[key]}"
            elif key == "SPACE":
                label = "Space"
            elif key == "<-":
                label = "delete"
            elif key == "Lang":
                label = "language"
            elif key == "Enter":
                label = "Enter"
            else:
                label = key

            cv2.putText(image, label, (x+10, y+row_height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            positions[key] = (x, y, x+key_width, y+row_height)
            x += key_width
        y += row_height
    return positions

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    key_positions = draw_keyboard(frame, current_key)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)
            cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)

            found_key = None
            for key, (x1, y1, x2, y2) in key_positions.items():
                if x1 < ix < x2 and y1 < iy < y2:
                    found_key = key
                    break

            if found_key:
                current_key = found_key
                dist = math.hypot(ix - tx, iy - ty)
                if dist < 20 and time.time() > cooldown: 
                    if current_key == "<-":
                        typed_text = typed_text[:-1]
                    elif current_key == "SPACE":
                        typed_text += " "
                    elif current_key == "Lang":
                        lang = "FA" if lang == "EN" else "EN"
                    elif current_key == "Enter":
                        with open("typed_text.txt", "a", encoding="utf-8") as f:
                            f.write(typed_text + "\n")
                        typed_text = ""  
                    else:
                        if lang == "FA" and current_key in persian_map:
                            typed_text += persian_map[current_key]
                        else:
                            typed_text += current_key
                    cooldown = time.time() + 0.6  

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, typed_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, f"Lang: {lang}", (1000, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Virtual Keyboard", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
