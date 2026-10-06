import cv2
import mediapipe as mp
import numpy as np
import time


# ============================================================
# HAND WHITEBOARD
# Index Finger Only Edition
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

WINDOW_NAME = "Hand Whiteboard"

# ------------------------------------------------------------
# Canvas / UI
# ------------------------------------------------------------

WHITE = (255, 255, 255)
DARK = (35, 35, 35)
GRAY = (120, 120, 120)
LIGHT_GRAY = (238, 238, 238)

# Smoothness:
# Higher = smoother but slightly more delayed
SMOOTHING = 0.78

# Toolbar
TOOLBAR_HEIGHT = 78

# How long finger must stay over a button
CLICK_DELAY = 0.45

# Maximum undo states
MAX_HISTORY = 40

# Camera preview
CAMERA_PREVIEW_WIDTH = 230


# ------------------------------------------------------------
# Colors - BGR
# ------------------------------------------------------------

COLORS = {
    "BLACK": (25, 25, 25),
    "RED": (55, 60, 225),
    "ORANGE": (30, 145, 245),
    "YELLOW": (40, 215, 245),
    "GREEN": (55, 185, 75),
    "BLUE": (220, 90, 45),
    "PURPLE": (185, 70, 185),
    "PINK": (220, 90, 175),
}


BRUSH_SIZES = [
    4,
    8,
    14,
    22,
    32,
]


# ------------------------------------------------------------
# MediaPipe
# ------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ============================================================
# HELPERS
# ============================================================

def lerp_point(old, new, smoothing):
    """
    Smooth hand movement.
    """

    if old is None:
        return new

    x = int(
        old[0] * smoothing
        + new[0] * (1 - smoothing)
    )

    y = int(
        old[1] * smoothing
        + new[1] * (1 - smoothing)
    )

    return x, y


def point_inside(point, rect):

    x, y = point

    x1, y1, x2, y2 = rect

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


def fingers_state(hand, handedness):

    """
    Returns:

    thumb
    index
    middle
    ring
    pinky
    """

    lm = hand.landmark

    # -------------------------
    # Thumb
    # -------------------------

    if handedness == "Right":

        thumb_up = (
            lm[4].x
            <
            lm[3].x
        )

    else:

        thumb_up = (
            lm[4].x
            >
            lm[3].x
        )

    # -------------------------
    # Other fingers
    # -------------------------

    index_up = (
        lm[8].y
        <
        lm[6].y
    )

    middle_up = (
        lm[12].y
        <
        lm[10].y
    )

    ring_up = (
        lm[16].y
        <
        lm[14].y
    )

    pinky_up = (
        lm[20].y
        <
        lm[18].y
    )

    return (
        thumb_up,
        index_up,
        middle_up,
        ring_up,
        pinky_up
    )


def only_index_up(state):

    thumb, index, middle, ring, pinky = state

    # We deliberately ignore the thumb.
    #
    # The important part is:
    # INDEX = UP
    # MIDDLE = DOWN
    # RING = DOWN
    # PINKY = DOWN

    return (
        index
        and
        not middle
        and
        not ring
        and
        not pinky
    )


# ============================================================
# TOOLBAR
# ============================================================

def create_toolbar(screen_width):

    items = {
        "colors": {},
        "sizes": {},
        "tools": {}
    }

    # --------------------------------------------------------
    # Color buttons
    # --------------------------------------------------------

    color_size = 42
    color_gap = 9

    color_names = list(COLORS.keys())

    total_color_width = (
        len(color_names)
        * color_size
        +
        (len(color_names) - 1)
        * color_gap
    )

    start_x = 28

    for name in color_names:

        items["colors"][name] = (
            start_x,
            18,
            start_x + color_size,
            60
        )

        start_x += (
            color_size
            + color_gap
        )

    # --------------------------------------------------------
    # Brush sizes
    # --------------------------------------------------------

    size_start = 405

    for size in BRUSH_SIZES:

        items["sizes"][str(size)] = (
            size_start,
            20,
            size_start + 42,
            58
        )

        size_start += 48

    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------

    tools_start = 665

    items["tools"]["ERASER"] = (
        tools_start,
        17,
        tools_start + 82,
        61
    )

    tools_start += 90

    items["tools"]["UNDO"] = (
        tools_start,
        17,
        tools_start + 62,
        61
    )

    tools_start += 70

    items["tools"]["REDO"] = (
        tools_start,
        17,
        tools_start + 62,
        61
    )

    tools_start += 70

    items["tools"]["CLEAR"] = (
        tools_start,
        17,
        tools_start + 70,
        61
    )

    tools_start += 78

    items["tools"]["SAVE"] = (
        tools_start,
        17,
        tools_start + 65,
        61
    )

    return items


# ============================================================
# DRAW TOOLBAR
# ============================================================

def draw_toolbar(
    frame,
    toolbar,
    selected_color,
    selected_size,
    eraser,
    hover_item=None
):

    h, w = frame.shape[:2]

    toolbar_width = min(
        1040,
        w - 30
    )

    x1 = (w - toolbar_width) // 2
    y1 = 12

    x2 = x1 + toolbar_width
    y2 = y1 + TOOLBAR_HEIGHT

    # --------------------------------------------------------
    # Shadow
    # --------------------------------------------------------

    shadow = frame.copy()

    cv2.rectangle(
        shadow,
        (x1 + 4, y1 + 5),
        (x2 + 4, y2 + 5),
        (170, 170, 170),
        -1
    )

    cv2.addWeighted(
        shadow,
        0.25,
        frame,
        0.75,
        0,
        frame
    )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (250, 250, 250),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (215, 215, 215),
        1
    )

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    for name, rect in toolbar["colors"].items():

        rx1, ry1, rx2, ry2 = rect

        center = (
            (rx1 + rx2) // 2,
            (ry1 + ry2) // 2
        )

        # Hover
        if hover_item == ("color", name):

            cv2.circle(
                frame,
                center,
                23,
                (190, 190, 190),
                2
            )

        # Selected
        if (
            selected_color == name
            and
            not eraser
        ):

            cv2.circle(
                frame,
                center,
                22,
                DARK,
                2
            )

        cv2.circle(
            frame,
            center,
            16,
            COLORS[name],
            -1
        )

    # --------------------------------------------------------
    # Brush Sizes
    # --------------------------------------------------------

    for size_text, rect in toolbar["sizes"].items():

        rx1, ry1, rx2, ry2 = rect

        size = int(size_text)

        center = (
            (rx1 + rx2) // 2,
            (ry1 + ry2) // 2
        )

        # Background
        if (
            selected_size == size
            and
            not eraser
        ):

            cv2.rectangle(
                frame,
                (rx1 - 2, ry1 - 2),
                (rx2 + 2, ry2 + 2),
                (220, 220, 220),
                -1
            )

            cv2.rectangle(
                frame,
                (rx1 - 2, ry1 - 2),
                (rx2 + 2, ry2 + 2),
                DARK,
                1
            )

        cv2.circle(
            frame,
            center,
            min(
                max(size // 2, 3),
                13
            ),
            DARK,
            -1
        )

    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------

    labels = {
        "ERASER": "ERASER",
        "UNDO": "UNDO",
        "REDO": "REDO",
        "CLEAR": "CLEAR",
        "SAVE": "SAVE"
    }

    for name, rect in toolbar["tools"].items():

        rx1, ry1, rx2, ry2 = rect

        active = (
            name == "ERASER"
            and
            eraser
        )

        hovered = (
            hover_item
            ==
            ("tool", name)
        )

        if active:

            bg = (215, 215, 215)

        elif hovered:

            bg = (228, 228, 228)

        else:

            bg = (242, 242, 242)

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            bg,
            -1
        )

        cv2.rectangle(
            frame,
            (rx1, ry1),
            (rx2, ry2),
            (205, 205, 205),
            1
        )

        text_size = cv2.getTextSize(
            labels[name],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            1
        )[0]

        tx = (
            rx1
            +
            (rx2 - rx1 - text_size[0])
            // 2
        )

        ty = (
            ry1
            +
            (ry2 - ry1 + text_size[1])
            // 2
        )

        cv2.putText(
            frame,
            labels[name],
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            DARK,
            1,
            cv2.LINE_AA
        )


# ============================================================
# TOOLBAR HIT TEST
# ============================================================

def get_toolbar_item(
    point,
    toolbar
):

    # Colors
    for name, rect in toolbar["colors"].items():

        if point_inside(
            point,
            rect
        ):

            return (
                "color",
                name
            )

    # Sizes
    for size, rect in toolbar["sizes"].items():

        if point_inside(
            point,
            rect
        ):

            return (
                "size",
                int(size)
            )

    # Tools
    for name, rect in toolbar["tools"].items():

        if point_inside(
            point,
            rect
        ):

            return (
                "tool",
                name
            )

    return None


# ============================================================
# CAMERA PREVIEW
# ============================================================

def draw_camera_preview(
    frame,
    preview
):

    h, w = preview.shape[:2]

    preview_w = CAMERA_PREVIEW_WIDTH

    preview_h = int(
        h
        *
        preview_w
        /
        w
    )

    preview = cv2.resize(
        preview,
        (
            preview_w,
            preview_h
        )
    )

    fh, fw = frame.shape[:2]

    x = (
        fw
        -
        preview_w
        -
        20
    )

    y = (
        fh
        -
        preview_h
        -
        20
    )

    # Background
    cv2.rectangle(
        frame,
        (
            x - 4,
            y - 4
        ),
        (
            x + preview_w + 4,
            y + preview_h + 4
        ),
        WHITE,
        -1
    )

    frame[
        y:y + preview_h,
        x:x + preview_w
    ] = preview

    cv2.rectangle(
        frame,
        (
            x - 3,
            y - 3
        ),
        (
            x + preview_w + 3,
            y + preview_h + 3
        ),
        DARK,
        2
    )


# ============================================================
# STATUS
# ============================================================

def draw_status(
    frame,
    status
):

    h, w = frame.shape[:2]

    text_size = cv2.getTextSize(
        status,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )[0]

    box_width = (
        text_size[0]
        +
        32
    )

    x1 = 20
    y1 = h - 55

    x2 = (
        x1
        +
        box_width
    )

    y2 = h - 18

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (248, 248, 248),
        -1
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (220, 220, 220),
        1
    )

    cv2.putText(
        frame,
        status,
        (
            x1 + 16,
            y1 + 24
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        DARK,
        1,
        cv2.LINE_AA
    )


# ============================================================
# HISTORY
# ============================================================

def add_history(
    history,
    board
):

    history.append(
        board.copy()
    )

    if len(history) > MAX_HISTORY:

        history.pop(0)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "ERROR: Camera could not be opened."
        )

        print(
            "Try changing CAMERA_INDEX = 0 to 1."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    ret, first_frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read camera."
        )

        cap.release()

        return

    first_frame = cv2.flip(
        first_frame,
        1
    )

    height, width = (
        first_frame.shape[:2]
    )

    # --------------------------------------------------------
    # White Canvas
    # --------------------------------------------------------

    board = np.full(
        (
            height,
            width,
            3
        ),
        WHITE,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []

    redo_history = []

    add_history(
        history,
        board
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    selected_color = "BLACK"

    selected_size = 8

    eraser = False

    previous_point = None

    smooth_point = None

    drawing = False

    status = "Ready"

    hover_item = None

    hover_start = 0

    last_click_item = None

    # --------------------------------------------------------
    # Toolbar
    # --------------------------------------------------------

    toolbar = create_toolbar(
        width
    )

    # --------------------------------------------------------
    # Fullscreen
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=1,

        model_complexity=1,

        min_detection_confidence=0.65,

        min_tracking_confidence=0.65

    ) as hands:

        while True:

            ret, camera = cap.read()

            if not ret:
                break

            # Mirror camera
            camera = cv2.flip(
                camera,
                1
            )

            h, w = camera.shape[:2]

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                camera,
                cv2.COLOR_BGR2RGB
            )

            rgb.flags.writeable = False

            results = hands.process(
                rgb
            )

            rgb.flags.writeable = True

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            preview = camera.copy()

            index_point = None

            index_active = False

            # ------------------------------------------------
            # Hand detected
            # ------------------------------------------------

            if results.multi_hand_landmarks:

                hand = (
                    results
                    .multi_hand_landmarks[0]
                )

                if results.multi_handedness:

                    handedness = (
                        results
                        .multi_handedness[0]
                        .classification[0]
                        .label
                    )

                else:

                    handedness = "Right"

                state = fingers_state(
                    hand,
                    handedness
                )

                # Only index finger
                index_active = only_index_up(
                    state
                )

                # ------------------------------------------------
                # Index fingertip
                # ------------------------------------------------

                raw_point = (
                    int(
                        hand.landmark[8].x
                        *
                        w
                    ),
                    int(
                        hand.landmark[8].y
                        *
                        h
                    )
                )

                smooth_point = lerp_point(
                    smooth_point,
                    raw_point,
                    SMOOTHING
                )

                index_point = (
                    smooth_point
                )

                # ------------------------------------------------
                # Draw hand skeleton only in preview
                # ------------------------------------------------

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

                # Highlight INDEX fingertip
                cv2.circle(
                    preview,
                    (
                        int(
                            hand.landmark[8].x
                            * w
                        ),
                        int(
                            hand.landmark[8].y
                            * h
                        )
                    ),
                    12,
                    (0, 255, 255),
                    2
                )

            else:

                smooth_point = None

                previous_point = None

                drawing = False

            # ====================================================
            # ONLY INDEX ACTIVE
            # ====================================================

            if (
                index_active
                and
                index_point is not None
            ):

                # ------------------------------------------------
                # Is finger over toolbar?
                # ------------------------------------------------

                item = get_toolbar_item(
                    index_point,
                    toolbar
                )

                # ------------------------------------------------
                # TOOLBAR
                # ------------------------------------------------

                if item is not None:

                    # Stop drawing
                    previous_point = None
                    drawing = False

                    # Hover
                    if item != hover_item:

                        hover_item = item

                        hover_start = (
                            time.time()
                        )

                        last_click_item = None

                    else:

                        # Finger remained on item
                        elapsed = (
                            time.time()
                            -
                            hover_start
                        )

                        if (
                            elapsed
                            >= CLICK_DELAY
                            and
                            last_click_item
                            != item
                        ):

                            last_click_item = item

                            item_type = item[0]
                            item_value = item[1]

                            # --------------------------------
                            # COLOR
                            # --------------------------------

                            if (
                                item_type
                                ==
                                "color"
                            ):

                                selected_color = (
                                    item_value
                                )

                                eraser = False

                                status = (
                                    "Color: "
                                    +
                                    item_value
                                )

                            # --------------------------------
                            # SIZE
                            # --------------------------------

                            elif (
                                item_type
                                ==
                                "size"
                            ):

                                selected_size = (
                                    item_value
                                )

                                eraser = False

                                status = (
                                    "Brush size: "
                                    +
                                    str(
                                        item_value
                                    )
                                )

                            # --------------------------------
                            # TOOLS
                            # --------------------------------

                            elif (
                                item_type
                                ==
                                "tool"
                            ):

                                # ERASER
                                if (
                                    item_value
                                    ==
                                    "ERASER"
                                ):

                                    eraser = True

                                    status = (
                                        "Eraser"
                                    )

                                # UNDO
                                elif (
                                    item_value
                                    ==
                                    "UNDO"
                                ):

                                    if (
                                        len(
                                            history
                                        )
                                        >
                                        1
                                    ):

                                        redo_history.append(
                                            history.pop()
                                        )

                                        board = (
                                            history[-1]
                                            .copy()
                                        )

                                        status = (
                                            "Undo"
                                        )

                                # REDO
                                elif (
                                    item_value
                                    ==
                                    "REDO"
                                ):

                                    if redo_history:

                                        board = (
                                            redo_history
                                            .pop()
                                        )

                                        add_history(
                                            history,
                                            board
                                        )

                                        status = (
                                            "Redo"
                                        )

                                # CLEAR
                                elif (
                                    item_value
                                    ==
                                    "CLEAR"
                                ):

                                    board = np.full(
                                        (
                                            height,
                                            width,
                                            3
                                        ),
                                        WHITE,
                                        dtype=np.uint8
                                    )

                                    add_history(
                                        history,
                                        board
                                    )

                                    redo_history.clear()

                                    status = (
                                        "Canvas cleared"
                                    )

                                # SAVE
                                elif (
                                    item_value
                                    ==
                                    "SAVE"
                                ):

                                    filename = (
                                        "whiteboard_"
                                        +
                                        str(
                                            int(
                                                time.time()
                                            )
                                        )
                                        +
                                        ".png"
                                    )

                                    cv2.imwrite(
                                        filename,
                                        board
                                    )

                                    status = (
                                        "Saved: "
                                        +
                                        filename
                                    )

                # ====================================================
                # DRAW ON CANVAS
                # ====================================================

                else:

                    hover_item = None
                    last_click_item = None

                    # Don't draw in toolbar
                    if (
                        index_point[1]
                        >
                        TOOLBAR_HEIGHT + 20
                    ):

                        if not drawing:

                            previous_point = (
                                index_point
                            )

                            drawing = True

                            # Save state BEFORE stroke
                            add_history(
                                history,
                                board
                            )

                            redo_history.clear()

                        else:

                            if (
                                previous_point
                                is not None
                            ):

                                # Eraser
                                if eraser:

                                    color = WHITE

                                    thickness = max(
                                        selected_size * 2,
                                        20
                                    )

                                # Pen
                                else:

                                    color = COLORS[
                                        selected_color
                                    ]

                                    thickness = (
                                        selected_size
                                    )

                                # ------------------------------------------------
                                # Draw line
                                # ------------------------------------------------

                                cv2.line(

                                    board,

                                    previous_point,

                                    index_point,

                                    color,

                                    thickness,

                                    cv2.LINE_AA
                                )

                                # Round cap
                                cv2.circle(

                                    board,

                                    index_point,

                                    thickness // 2,

                                    color,

                                    -1,

                                    cv2.LINE_AA
                                )

                                previous_point = (
                                    index_point
                                )

                                status = (
                                    "Erasing"
                                    if eraser
                                    else
                                    "Drawing"
                                )

                    else:

                        previous_point = None

                        drawing = False

            # ====================================================
            # INDEX NOT ACTIVE
            # ====================================================

            else:

                previous_point = None

                drawing = False

                hover_item = None

                last_click_item = None

            # ====================================================
            # OUTPUT
            # ====================================================

            output = board.copy()

            # ----------------------------------------------------
            # Toolbar
            # ----------------------------------------------------

            draw_toolbar(

                output,

                toolbar,

                selected_color,

                selected_size,

                eraser,

                hover_item
            )

            # ----------------------------------------------------
            # Camera preview
            # ----------------------------------------------------

            draw_camera_preview(
                output,
                preview
            )

            # ----------------------------------------------------
            # Finger cursor
            # ----------------------------------------------------

            if index_point is not None:

                if eraser:

                    cursor_color = (
                        100,
                        100,
                        100
                    )

                else:

                    cursor_color = COLORS[
                        selected_color
                    ]

                cursor_radius = max(
                    6,
                    selected_size // 2
                )

                # Outer white ring
                cv2.circle(
                    output,
                    index_point,
                    cursor_radius + 7,
                    WHITE,
                    2
                )

                # Current tool color
                cv2.circle(
                    output,
                    index_point,
                    cursor_radius,
                    cursor_color,
                    2
                )

                # Center
                cv2.circle(
                    output,
                    index_point,
                    2,
                    cursor_color,
                    -1
                )

            # ----------------------------------------------------
            # Status
            # ----------------------------------------------------

            draw_status(
                output,
                status
            )

            # ----------------------------------------------------
            # Instruction
            # ----------------------------------------------------

            instruction = (
                "INDEX FINGER: Draw / Select"
            )

            text_size = cv2.getTextSize(
                instruction,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1
            )[0]

            cv2.putText(
                output,
                instruction,
                (
                    width
                    -
                    text_size[0]
                    -
                    20,
                    height
                    -
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (145, 145, 145),
                1,
                cv2.LINE_AA
            )

            # ====================================================
            # SHOW
            # ====================================================

            cv2.imshow(
                WINDOW_NAME,
                output
            )

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )

            # ----------------------------------------------------
            # Keyboard shortcuts
            # ----------------------------------------------------

            # Exit
            if key in (
                ord("q"),
                27
            ):

                break

            # Clear
            elif key == ord("c"):

                board = np.full(
                    (
                        height,
                        width,
                        3
                    ),
                    WHITE,
                    dtype=np.uint8
                )

                add_history(
                    history,
                    board
                )

                redo_history.clear()

                status = (
                    "Canvas cleared"
                )

            # Save
            elif key == ord("s"):

                filename = (
                    "whiteboard_"
                    +
                    str(
                        int(
                            time.time()
                        )
                    )
                    +
                    ".png"
                )

                cv2.imwrite(
                    filename,
                    board
                )

                status = (
                    "Saved: "
                    +
                    filename
                )

            # Undo
            elif key == ord("z"):

                if (
                    len(history)
                    >
                    1
                ):

                    redo_history.append(
                        history.pop()
                    )

                    board = (
                        history[-1]
                        .copy()
                    )

                    status = (
                        "Undo"
                    )

            # Redo
            elif key == ord("y"):

                if redo_history:

                    board = (
                        redo_history
                        .pop()
                    )

                    add_history(
                        history,
                        board
                    )

                    status = (
                        "Redo"
                    )

            # Eraser
            elif key == ord("e"):

                eraser = True

                status = (
                    "Eraser"
                )

            # Pen
            elif key == ord("p"):

                eraser = False

                status = (
                    "Pen: "
                    +
                    selected_color
                )

            # Brush size
            elif key in (
                ord("1"),
                ord("2"),
                ord("3"),
                ord("4"),
                ord("5")
            ):

                index = (
                    int(
                        chr(key)
                    )
                    -
                    1
                )

                if (
                    index
                    <
                    len(
                        BRUSH_SIZES
                    )
                ):

                    selected_size = (
                        BRUSH_SIZES[
                            index
                        ]
                    )

                    eraser = False

                    status = (
                        "Brush size: "
                        +
                        str(
                            selected_size
                        )
                    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()