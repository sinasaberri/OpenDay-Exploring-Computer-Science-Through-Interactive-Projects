"""Virtual Keyboard — type in the air with your fingers.

An on-screen keyboard is drawn over the camera feed. Hovering the index
fingertip over a key highlights it; pinching index and thumb together
"presses" the key (with a short debounce so one pinch types one key).
The keyboard supports English and Persian output, backspace, space and
enter. Pressing Enter appends the typed line to ``typed_text.txt`` next
to this script.

Run:  python "Air_Keyboard/Virtual Keyboard.py"
Keys: ESC or Q to quit.
"""

import math
import os
import time

import arabic_reshaper
import cv2
import mediapipe as mp
import numpy as np
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TYPED_TEXT_PATH = os.path.join(BASE_DIR, "typed_text.txt")

PINCH_RATIO = 0.045       # pinch distance as a fraction of frame height
SMOOTHING = 0.5           # fraction of the new fingertip position kept
DEBOUNCE_SECONDS = 0.6
MAX_CAMERA_FAILURES = 90  # ~3 s of dead frames before giving up

PERSIAN_MAP = {
    "Q": "ض", "W": "ص", "E": "ث", "R": "ق", "T": "ف", "Y": "غ", "U": "ع", "I": "ه", "O": "خ", "P": "ح",
    "A": "ش", "S": "س", "D": "ی", "F": "ب", "G": "ل", "H": "ا", "J": "ت", "K": "ن", "L": "م",
    "Z": "ظ", "X": "ط", "C": "ز", "V": "ر", "B": "ذ", "N": "د", "M": "پ",
}

KEYBOARD_ROWS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "<-", "Lang", "Enter"],
]
KEY_SLOT_WIDTHS = {"SPACE": 4, "<-": 2, "Lang": 2, "Enter": 2}  # letters = 1 slot

KEY_COLOR = (255, 0, 0)
HOVER_COLOR = (0, 255, 255)
PRESSED_COLOR = (0, 255, 0)
INDEX_DOT_COLOR = (0, 255, 255)
THUMB_DOT_COLOR = (0, 0, 255)


def open_camera(index=0, width=1280, height=720):
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


def open_persian_font(size):
    """Find a system font that can show Persian letters (falls back silently)."""
    for name in ("arial.ttf", "tahoma.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def shape_persian(text):
    """Reshape and reorder Persian text so it renders correctly with PIL."""
    reshaped = arabic_reshaper.reshape(text)
    try:
        return get_display(reshaped)
    except Exception:
        return reshaped


def has_persian(text):
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def draw_persian_overlay(image, texts, font):
    """Draw (org, text) pairs with real Persian shaping; returns a new BGR image."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    for (x, y), text in texts:
        draw.text((x, y), shape_persian(text), font=font, fill=(200, 0, 0))
    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def compute_key_rects(frame_height, frame_width):
    """Return {key: (x1, y1, x2, y2)} and [(org, persian_char), ...] labels."""
    positions = {}
    persian_labels = []

    row_height = frame_height // (len(KEYBOARD_ROWS) + 2)
    y = row_height * 2

    for row in KEYBOARD_ROWS:
        widths = [KEY_SLOT_WIDTHS.get(key, 1) for key in row]
        unit = frame_width // sum(widths)
        x = (frame_width - unit * sum(widths)) // 2
        for key, key_width_slots in zip(row, widths):
            key_width = unit * key_width_slots
            positions[key] = (x, y, x + key_width, y + row_height)
            if key in PERSIAN_MAP:
                persian_labels.append(((x + key_width // 2, y + 12), PERSIAN_MAP[key]))
            x += key_width
        y += row_height
    return positions, persian_labels


def draw_keyboard(image, positions, lang, hovered_key, pressed_key):
    """Draw the keys and their labels onto the image."""
    for key, (x1, y1, x2, y2) in positions.items():
        color = KEY_COLOR
        if pressed_key == key:
            color = PRESSED_COLOR
        elif hovered_key == key:
            color = HOVER_COLOR

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        if key == "SPACE":
            label = "Space"
        elif key == "<-":
            label = "delete"
        elif key == "Lang":
            label = "language"
        else:
            label = key
        cv2.putText(image, label, (x1 + 10, y2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def apply_key(key, typed_text, lang):
    """Return (typed_text, lang) after pressing a key."""
    if key == "<-":
        return typed_text[:-1], lang
    if key == "SPACE":
        return typed_text + " ", lang
    if key == "Lang":
        return typed_text, "FA" if lang == "EN" else "EN"
    if key == "Enter":
        try:
            with open(TYPED_TEXT_PATH, "a", encoding="utf-8") as f:
                f.write(typed_text + "\n")
        except OSError as exc:
            print(f"Could not save typed text: {exc}")
        return "", lang
    if lang == "FA" and key in PERSIAN_MAP:
        return typed_text + PERSIAN_MAP[key], lang
    return typed_text + key, lang


def main():
    cap = open_camera()
    hands = mp.solutions.hands.Hands()

    typed_text = ""
    lang = "EN"
    cooldown = 0.0
    pressed_key = None      # key shown green until the debounce expires
    smooth_ix, smooth_iy = None, None
    camera_failures = 0
    persian_font = open_persian_font(28)

    cv2.namedWindow("Virtual Keyboard", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                camera_failures += 1
                if camera_failures > MAX_CAMERA_FAILURES:
                    print("Camera keeps failing — closing.")
                    break
                continue
            camera_failures = 0

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            key_positions, persian_labels = compute_key_rects(h, w)

            pressed_key = pressed_key if time.time() < cooldown else None
            hovered_key = None
            pinch_threshold = PINCH_RATIO * h
            pinching = False

            if results.multi_hand_landmarks:
                # Draw every hand, but only steer the keyboard with the first one
                # so a second hand (or a bystander) cannot type by accident.
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                landmarks = results.multi_hand_landmarks[0].landmark
                index_tip = landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]

                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Smooth the fingertip position so the highlight does not jitter.
                if smooth_ix is None:
                    smooth_ix, smooth_iy = ix, iy
                else:
                    smooth_ix = int(SMOOTHING * ix + (1 - SMOOTHING) * smooth_ix)
                    smooth_iy = int(SMOOTHING * iy + (1 - SMOOTHING) * smooth_iy)

                for key, (x1, y1, x2, y2) in key_positions.items():
                    if x1 < smooth_ix < x2 and y1 < smooth_iy < y2:
                        hovered_key = key
                        break

                pinch_dist = math.hypot(smooth_ix - tx, smooth_iy - ty)
                pinching = pinch_dist < pinch_threshold

                cv2.circle(frame, (smooth_ix, smooth_iy), 10, INDEX_DOT_COLOR, -1)
                cv2.circle(frame, (tx, ty), 10, THUMB_DOT_COLOR, -1)
                if pinching:
                    cv2.circle(frame, ((smooth_ix + tx) // 2, (smooth_iy + ty) // 2),
                               int(pinch_threshold), PRESSED_COLOR, 2)
            else:
                smooth_ix, smooth_iy = None, None

            draw_keyboard(frame, key_positions, lang, hovered_key, pressed_key)

            if pinching and hovered_key and time.time() > cooldown:
                typed_text, lang = apply_key(hovered_key, typed_text, lang)
                cooldown = time.time() + DEBOUNCE_SECONDS
                pressed_key = hovered_key

            # Real Persian glyphs cannot be drawn by cv2.putText, so shape and
            # draw them (key labels in FA mode and the typed text) with PIL.
            overlay_texts = list(persian_labels) if lang == "FA" else []
            if typed_text:
                overlay_texts.append(((50, 100), typed_text))
            if overlay_texts and (lang == "FA" or has_persian(typed_text)):
                frame = draw_persian_overlay(frame, overlay_texts, persian_font)
            elif typed_text:
                cv2.putText(frame, typed_text, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.putText(frame, f"Lang: {lang}", (w - 200, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "ESC / Q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Virtual Keyboard", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if cv2.getWindowProperty("Virtual Keyboard", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
