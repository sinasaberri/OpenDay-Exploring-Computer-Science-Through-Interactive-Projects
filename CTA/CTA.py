"""Catch the falling fruit — with your face!

A pygame game where a basket at the bottom of the screen follows the
horizontal position of your face, detected with an OpenCV Haar cascade.
Catch fruit for points (+1/+2/+3 by color); fruit that hits the ground
costs the same points. If no face is found for a couple of seconds the
basket gently drifts back to the center.

Run:  python CTA/CTA.py
Exit: close the window.
"""

import os
import random
import sys
import tempfile
import time

import cv2
import pygame

WIDTH, HEIGHT = 400, 800
FPS = 30

APPLE_SIZE = 30
APPLE_MIN_SPEED = 3
APPLE_MAX_SPEED = 5
SPAWN_EVERY_FRAMES = 60  # one fruit every ~2 seconds at 30 FPS

MOUTH_W, MOUTH_H = 100, 40
FACE_SMOOTHING = 0.15    # fraction of the face position applied per frame
CENTER_DRIFT = 0.10      # drift speed back to center when no face is seen
FACE_LOST_TIMEOUT = 2    # seconds without a face before drifting to center

MAX_CAMERA_FAILURES = 90  # ~3 s of dead frames before giving up

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)

FRUITS = [
    {"color": RED, "score": 1},
    {"color": ORANGE, "score": 2},
    {"color": YELLOW, "score": 3},
]


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam (index {}). Connect a camera and make sure "
            "no other program is using it.".format(index)
        )
    return cap


def load_face_cascade():
    """Load the built-in face cascade.

    OpenCV's C++ file loader cannot open paths containing non-ASCII
    characters (e.g. an em dash or accented letters in the project folder
    on Windows), so fall back to loading the bundled file from a temp path.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if cascade_path.isascii():
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            return cascade
        raise RuntimeError("Could not load the face detection cascade file.")
    # Non-ASCII install path: OpenCV's C++ loader cannot open the file, so
    # copy it via Python and load it from an ASCII-safe temp path instead.
    try:
        with open(cascade_path, "rb") as f:
            data = f.read()
        fd, temp_path = tempfile.mkstemp(suffix=".xml")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            cascade = cv2.CascadeClassifier(temp_path)
        finally:
            os.remove(temp_path)
    except OSError as exc:
        raise RuntimeError(f"Could not read the face cascade file: {exc}") from exc
    if cascade.empty():
        raise RuntimeError("Could not load the face detection cascade file.")
    return cascade


def add_apple(apples):
    fruit = random.choice(FRUITS)
    apples.append({
        "x": random.randint(WIDTH // 4, (WIDTH * 3) // 4 - APPLE_SIZE),
        "y": -APPLE_SIZE,
        "speed": random.randint(APPLE_MIN_SPEED, APPLE_MAX_SPEED),
        "color": fruit["color"],
        "score": fruit["score"],
    })


def track_face(cap, cascade, mouth_x, last_seen_time):
    """Update the basket position from the camera frame.

    Returns (mouth_x, last_seen_time, failed). ``failed`` is True when the
    camera itself stops delivering frames.
    """
    ret, frame = cap.read()
    if not ret:
        return mouth_x, last_seen_time, True

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]  # one player at a time: use the first detection
        face_center = int(x + w / 2)
        mouth_x = int((1 - FACE_SMOOTHING) * mouth_x + FACE_SMOOTHING * face_center)
        last_seen_time = time.time()
    elif time.time() - last_seen_time > FACE_LOST_TIMEOUT:
        mouth_x = int((1 - CENTER_DRIFT) * mouth_x + CENTER_DRIFT * (WIDTH // 2))
    return mouth_x, last_seen_time, False


def run_game(cap, cascade):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Catch The Apples")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 42)

    apples = []
    basket_fruits = []
    score = 0
    collected = 0
    frame_count = 0
    camera_failures = 0

    mouth_x = WIDTH // 2
    last_seen_time = time.time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        frame_count += 1
        if frame_count % SPAWN_EVERY_FRAMES == 0:
            add_apple(apples)

        for apple in apples:
            apple["y"] += apple["speed"]
            pygame.draw.circle(
                screen, apple["color"],
                (apple["x"] + APPLE_SIZE // 2, apple["y"] + APPLE_SIZE // 2),
                APPLE_SIZE // 2,
            )

        mouth_x, last_seen_time, camera_failed = track_face(
            cap, cascade, mouth_x, last_seen_time
        )
        if camera_failed:
            camera_failures += 1
            if camera_failures > MAX_CAMERA_FAILURES:
                print("Camera keeps failing — closing the game.")
                running = False
        else:
            camera_failures = 0

        basket_y = HEIGHT - 60
        basket_rect = (mouth_x - MOUTH_W // 2, basket_y, MOUTH_W, MOUTH_H)
        pygame.draw.rect(screen, BROWN, basket_rect)

        for apple in apples[:]:
            if (basket_rect[0] < apple["x"] < basket_rect[0] + basket_rect[2] and
                    basket_rect[1] < apple["y"] < basket_rect[1] + basket_rect[3]):
                score += apple["score"]
                collected += 1
                basket_fruits.append(apple["color"])
                apples.remove(apple)
            elif apple["y"] > HEIGHT:
                score -= apple["score"]
                apples.remove(apple)

        bx = basket_rect[0] + 5
        by = basket_rect[1] + 5
        for c in basket_fruits[-5:]:
            pygame.draw.circle(screen, c, (bx, by), 8)
            bx += 15

        screen.blit(font.render(f"Score: {score}", True, BLACK), (10, 10))
        screen.blit(font.render(f"Collected: {collected}", True, BLACK), (10, 50))

        pygame.display.update()
        clock.tick(FPS)


def main():
    cap = None
    try:
        cap = open_camera()
        cascade = load_face_cascade()
        pygame.init()
        run_game(cap, cascade)
    except RuntimeError as exc:
        print(exc)
        sys.exit(1)
    finally:
        if cap is not None:
            cap.release()
        pygame.quit()


if __name__ == "__main__":
    main()
