import cv2
import mediapipe as mp
import numpy as np
import math
import time


# ============================================================
# HAND WHITEBOARD
# Python + OpenCV + MediaPipe
#
# Gestures:
#   Index finger only       -> Draw
#   Index + middle fingers  -> Select toolbar
#   Fist                    -> Stop drawing
#
# Keyboard:
#   Q / ESC -> Exit
#   C       -> Clear
#   E       -> Eraser
#   P       -> Pen
#   S       -> Save
#   1-4     -> Brush size
# ============================================================


CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

WHITE = (255, 255, 255)

COLORS = {
    "BLACK": (25, 25, 25),
    "RED": (50, 50, 220),
    "GREEN": (50, 190, 70),
    "BLUE": (220, 80, 50),
    "YELLOW": (40, 210, 240),
    "PURPLE": (180, 70, 180),
}

BRUSH_SIZES = [4, 8, 14, 22]


# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def is_inside(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect

    return (
        x1 <= x <= x2 and
        y1 <= y <= y2
    )


def fingers_up(hand_landmarks, handedness):
    """
    Returns:

    [thumb, index, middle, ring, pinky]

    1 = finger is up
    0 = finger is down
    """

    lm = hand_landmarks.landmark

    fingers = []

    # Thumb
    if handedness == "Right":
        thumb_up = lm[4].x < lm[3].x
    else:
        thumb_up = lm[4].x > lm[3].x

    fingers.append(1 if thumb_up else 0)

    # Index, Middle, Ring, Pinky
    for tip, pip in [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18)
    ]:

        if lm[tip].y < lm[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


# ------------------------------------------------------------
# Toolbar
# ------------------------------------------------------------

def create_toolbar():

    toolbar = {
        "height": 90,
        "colors": {},
        "sizes": {},
        "tools": {}
    }

    # Colors
    x = 20

    for name in COLORS:

        toolbar["colors"][name] = (
            x,
            15,
            x + 50,
            65
        )

        x += 60

    # Brush sizes
    x = 340

    for size in BRUSH_SIZES:

        toolbar["sizes"][str(size)] = (
            x,
            15,
            x + 45,
            65
        )

        x += 55

    # Tools
    toolbar["tools"]["ERASER"] = (
        570,
        15,
        680,
        65
    )

    toolbar["tools"]["CLEAR"] = (
        690,
        15,
        790,
        65
    )

    toolbar["tools"]["SAVE"] = (
        800,
        15,
        900,
        65
    )

    toolbar["tools"]["EXIT"] = (
        910,
        15,
        1000,
        65
    )

    return toolbar


def draw_toolbar(
    frame,
    toolbar,
    selected_color,
    selected_size,
    eraser
):

    # Toolbar background
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], toolbar["height"]),
        (245, 245, 245),
        -1
    )

    # Separator
    cv2.line(
        frame,
        (0, toolbar["height"]),
        (frame.shape[1], toolbar["height"]),
        (180, 180, 180),
        2
    )

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    for name, rect in toolbar["colors"].items():

        x1, y1, x2, y2 = rect

        color = COLORS[name]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            -1
        )

        if name == selected_color and not eraser:

            cv2.rectangle(
                frame,
                (x1 - 3, y1 - 3),
                (x2 + 3, y2 + 3),
                (0, 0, 0),
                3
            )

    # --------------------------------------------------------
    # Brush sizes
    # --------------------------------------------------------

    for size_text, rect in toolbar["sizes"].items():

        x1, y1, x2, y2 = rect

        size = int(size_text)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (225, 225, 225),
            -1
        )

        if size == selected_size and not eraser:

            cv2.rectangle(
                frame,
                (x1 - 3, y1 - 3),
                (x2 + 3, y2 + 3),
                (0, 0, 0),
                3
            )

        center = (
            (x1 + x2) // 2,
            (y1 + y2) // 2
        )

        cv2.circle(
            frame,
            center,
            max(2, min(size // 2, 12)),
            (30, 30, 30),
            -1
        )

    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------

    tool_colors = {
        "ERASER": (150, 150, 150),
        "CLEAR": (80, 80, 80),
        "SAVE": (70, 150, 70),
        "EXIT": (70, 70, 180)
    }

    for name, rect in toolbar["tools"].items():

        x1, y1, x2, y2 = rect

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            tool_colors[name],
            -1
        )

        if name == "ERASER" and eraser:

            cv2.rectangle(
                frame,
                (x1 - 3, y1 - 3),
                (x2 + 3, y2 + 3),
                (0, 0, 0),
                3
            )

        cv2.putText(
            frame,
            name,
            (x1 + 7, y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

def draw_status(frame, text):

    h, w = frame.shape[:2]

    cv2.rectangle(
        frame,
        (10, h - 45),
        (500, h - 10),
        (245, 245, 245),
        -1
    )

    cv2.putText(
        frame,
        text,
        (20, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 40, 40),
        1,
        cv2.LINE_AA
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():

        print("Could not open webcam.")
        print("Try changing CAMERA_INDEX to 1.")

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # First frame
    ret, frame = cap.read()

    if not ret:

        print("Could not read webcam.")

        cap.release()

        return

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # Whiteboard
    board = np.full(
        (height, width, 3),
        WHITE,
        dtype=np.uint8
    )

    toolbar = create_toolbar()

    # Current settings
    selected_color = "BLACK"
    selected_size = 8

    eraser = False

    # Previous drawing position
    previous_point = None

    # Prevent multiple toolbar actions
    last_action_time = 0
    action_delay = 0.45

    status = "Show your hand"

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=1,

        model_complexity=1,

        min_detection_confidence=0.55,

        min_tracking_confidence=0.55

    ) as hands:

        while True:

            ret, camera = cap.read()

            if not ret:
                break

            # Mirror camera
            camera = cv2.flip(camera, 1)

            h, w = camera.shape[:2]

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                camera,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            results = hands.process(rgb)

            rgb.flags.writeable = True

            preview = camera.copy()

            current_gesture = "NO HAND"

            index_point = None

            # ------------------------------------------------
            # Hand detected
            # ------------------------------------------------

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                if results.multi_handedness:

                    handedness = (
                        results
                        .multi_handedness[0]
                        .classification[0]
                        .label
                    )

                else:

                    handedness = "Right"

                fingers = fingers_up(
                    hand,
                    handedness
                )

                thumb = fingers[0]
                index = fingers[1]
                middle = fingers[2]
                ring = fingers[3]
                pinky = fingers[4]

                # Index fingertip
                ix = int(
                    hand.landmark[8].x * w
                )

                iy = int(
                    hand.landmark[8].y * h
                )

                index_point = (ix, iy)

                # Draw hand skeleton
                mp_draw.draw_landmarks(

                    preview,

                    hand,

                    mp_hands.HAND_CONNECTIONS,

                    mp_draw.DrawingSpec(
                        color=(0, 180, 0),
                        thickness=2,
                        circle_radius=2
                    ),

                    mp_draw.DrawingSpec(
                        color=(0, 120, 255),
                        thickness=2
                    )
                )

                # ------------------------------------------------
                # Gesture
                # ------------------------------------------------

                if index == 1 and middle == 0:

                    current_gesture = "DRAW"

                elif index == 1 and middle == 1 and ring == 0:

                    current_gesture = "SELECT"

                else:

                    current_gesture = "IDLE"

                # Cursor
                cv2.circle(
                    preview,
                    index_point,
                    10,
                    (0, 0, 0),
                    2
                )

                cv2.circle(
                    preview,
                    index_point,
                    4,
                    (0, 255, 255),
                    -1
                )

                # ------------------------------------------------
                # SELECT MODE
                # ------------------------------------------------

                if current_gesture == "SELECT":

                    previous_point = None

                    if index_point[1] <= toolbar["height"]:

                        now = time.time()

                        # Colors
                        for name, rect in toolbar["colors"].items():

                            if is_inside(
                                index_point,
                                rect
                            ):

                                if (
                                    now - last_action_time
                                    > action_delay
                                ):

                                    selected_color = name

                                    eraser = False

                                    status = (
                                        f"Color: {name}"
                                    )

                                    last_action_time = now

                        # Brush sizes
                        for size_text, rect in toolbar["sizes"].items():

                            if is_inside(
                                index_point,
                                rect
                            ):

                                if (
                                    now - last_action_time
                                    > action_delay
                                ):

                                    selected_size = int(
                                        size_text
                                    )

                                    eraser = False

                                    status = (
                                        f"Brush: {selected_size}"
                                    )

                                    last_action_time = now

                        # Tools
                        for name, rect in toolbar["tools"].items():

                            if is_inside(
                                index_point,
                                rect
                            ):

                                if (
                                    now - last_action_time
                                    > action_delay
                                ):

                                    last_action_time = now

                                    # Eraser
                                    if name == "ERASER":

                                        eraser = True

                                        status = "Eraser selected"

                                    # Clear
                                    elif name == "CLEAR":

                                        board[:] = WHITE

                                        status = "Board cleared"

                                    # Save
                                    elif name == "SAVE":

                                        filename = (
                                            f"whiteboard_"
                                            f"{int(time.time())}.png"
                                        )

                                        cv2.imwrite(
                                            filename,
                                            board
                                        )

                                        status = (
                                            f"Saved: {filename}"
                                        )

                                    # Exit
                                    elif name == "EXIT":

                                        cap.release()

                                        cv2.destroyAllWindows()

                                        return

                # ------------------------------------------------
                # DRAW MODE
                # ------------------------------------------------

                elif current_gesture == "DRAW":

                    # Don't draw on toolbar
                    if index_point[1] > toolbar["height"]:

                        if previous_point is not None:

                            if eraser:

                                color = WHITE

                                thickness = max(
                                    selected_size * 2,
                                    18
                                )

                            else:

                                color = COLORS[
                                    selected_color
                                ]

                                thickness = selected_size

                            # Main line
                            cv2.line(

                                board,

                                previous_point,

                                index_point,

                                color,

                                thickness,

                                cv2.LINE_AA
                            )

                            # Round line ending
                            cv2.circle(

                                board,

                                index_point,

                                thickness // 2,

                                color,

                                -1,

                                cv2.LINE_AA
                            )

                        previous_point = index_point

                        if eraser:

                            status = "ERASER"

                        else:

                            status = (
                                f"DRAWING - "
                                f"{selected_color}"
                            )

                    else:

                        previous_point = None

                else:

                    previous_point = None

            else:

                previous_point = None

            # ----------------------------------------------------
            # Final image
            # ----------------------------------------------------

            output = board.copy()

            # Toolbar
            draw_toolbar(
                output,
                toolbar,
                selected_color,
                selected_size,
                eraser
            )

            # ----------------------------------------------------
            # Camera preview
            # ----------------------------------------------------

            preview_width = 280

            preview_height = int(
                preview.shape[0]
                * preview_width
                / preview.shape[1]
            )

            preview_small = cv2.resize(
                preview,
                (
                    preview_width,
                    preview_height
                )
            )

            px1 = width - preview_width - 15
            py1 = height - preview_height - 15

            px2 = width - 15
            py2 = height - 15

            cv2.rectangle(
                output,
                (px1 - 3, py1 - 3),
                (px2 + 3, py2 + 3),
                (30, 30, 30),
                3
            )

            output[
                py1:py2,
                px1:px2
            ] = preview_small

            # Status
            draw_status(
                output,
                f"{status} | {current_gesture}"
            )

            # Title
            cv2.putText(
                output,
                "HAND WHITEBOARD",
                (width - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 50, 50),
                2,
                cv2.LINE_AA
            )

            # ----------------------------------------------------
            # Show
            # ----------------------------------------------------

            cv2.imshow(
                "Hand Whiteboard",
                output
            )

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord("q") or key == 27:

                break

            # Clear
            elif key == ord("c"):

                board[:] = WHITE

                status = "Board cleared"

            # Eraser
            elif key == ord("e"):

                eraser = True

                status = "Eraser selected"

            # Pen
            elif key == ord("p"):

                eraser = False

                status = (
                    f"Pen: {selected_color}"
                )

            # Save
            elif key == ord("s"):

                filename = (
                    f"whiteboard_"
                    f"{int(time.time())}.png"
                )

                cv2.imwrite(
                    filename,
                    board
                )

                status = (
                    f"Saved: {filename}"
                )

            # Brush sizes
            elif key in [
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4")
            ]:

                index = int(
                    chr(key)
                ) - 1

                selected_size = BRUSH_SIZES[index]

                eraser = False

                status = (
                    f"Brush: {selected_size}"
                )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()