"""Two-player air Pong — move the paddles with your hands.

Each player holds up a hand; the vertical position of the index fingertip
controls the paddle on that side of the screen (leftmost hand = left
paddle, rightmost hand = right paddle). First player to reach 5 points
wins. The ball serves after a short countdown and gets faster with every
point, up to a cap, so rounds stay exciting during a live demo.

Run:  python PingPong/PingPong.py
Keys: ENTER (after game over) restarts, close the window to quit.
"""

import random
import time

import cv2
import mediapipe as mp
import pygame

WIDTH, HEIGHT = 1280, 800
BALL_SIZE = 30
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 120
INITIAL_BALL_SPEED = 10
MAX_BALL_SPEED = 15
SPEED_INCREMENT = 2
FPS = 60
PAUSE_DURATION = 2  # seconds between a point and the next serve
WINNING_SCORE = 5

BALL_ORANGE = pygame.Color("#F97C00")
PADDLE_BLUE = pygame.Color("#E2DFD0")
BACKGROUND_GRAY = pygame.Color("#0C0C0C")
SCORE_GOLD = (255, 215, 0)

PLAYER1_NAME = "A"
PLAYER2_NAME = "B"

INDEX_TIP = 8  # MediaPipe landmark used to steer a paddle


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


def draw_scores(screen, font, score1, score2):
    score1_text = font.render(f"{PLAYER1_NAME} : {score1}", True, SCORE_GOLD)
    score2_text = font.render(f"{PLAYER2_NAME} : {score2}", True, SCORE_GOLD)
    screen.blit(score1_text, (10, 10))
    screen.blit(score2_text, (WIDTH - score2_text.get_width() - 10, 10))


def reset_ball(ball, ball_speed):
    """Center the ball, pause the round, and pick a random serve direction."""
    global ball_paused, pause_start_time
    ball.center = (WIDTH // 2, HEIGHT // 2)
    ball_paused = True
    pause_start_time = time.time()
    return ball_speed * random.choice((1, -1)), ball_speed * random.choice((1, -1))


def update_paddles(results, paddle1, paddle2):
    """Map hands to paddles by fingertip X and smooth their Y positions."""
    if not (results and results.multi_hand_landmarks):
        return

    hands = []
    for hand_landmarks in results.multi_hand_landmarks:
        tip = hand_landmarks.landmark[INDEX_TIP]
        hands.append((tip.x, tip.y * HEIGHT))
    hands.sort(key=lambda hand: hand[0])  # leftmost fingertip -> paddle 1

    targets = {}
    for i, (hand_x, hand_y) in enumerate(hands[:2]):
        # One hand only: keep the original behavior and steer the paddle
        # on the side of the screen the hand is on.
        paddle = i if len(hands) > 1 else (0 if hand_x < 0.5 else 1)
        targets[paddle] = hand_y - PADDLE_HEIGHT // 2

    paddle1.y = int(0.7 * paddle1.y + 0.3 * targets.get(0, paddle1.y))
    paddle2.y = int(0.7 * paddle2.y + 0.3 * targets.get(1, paddle2.y))
    paddle1.y = max(0, min(HEIGHT - PADDLE_HEIGHT, paddle1.y))
    paddle2.y = max(0, min(HEIGHT - PADDLE_HEIGHT, paddle2.y))


def run_game(hands, cap):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Ping Pong")
    clock = pygame.time.Clock()
    score_font = pygame.font.Font(None, 36)
    game_over_font = pygame.font.Font(None, 72)

    ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2,
                       BALL_SIZE, BALL_SIZE)
    paddle1 = pygame.Rect(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
    paddle2 = pygame.Rect(WIDTH - 60, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

    ball_speed = INITIAL_BALL_SPEED
    global ball_paused, pause_start_time
    ball_paused = True
    pause_start_time = time.time()
    ball_dx, ball_dy = reset_ball(ball, ball_speed)
    ball_in_motion = False

    score1, score2 = 0, 0
    game_over = False

    running = True
    while running:
        ret, frame = cap.read()
        if ret:
            results = hands.process(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
        else:
            results = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return True

        if not game_over:
            update_paddles(results, paddle1, paddle2)

            if ball_paused and time.time() - pause_start_time >= PAUSE_DURATION:
                ball_paused = False
                ball_in_motion = True

            if ball_in_motion:
                ball.x += ball_dx
                ball.y += ball_dy

            if ball.top <= 0 or ball.bottom >= HEIGHT:
                ball_dy *= -1

            # Mirror the incoming horizontal angle so rallies stay playable.
            if ball.colliderect(paddle1):
                ball_dx = abs(ball_dx)
            elif ball.colliderect(paddle2):
                ball_dx = -abs(ball_dx)

            if ball.left <= 0:
                score2 += 1
                ball_speed = min(ball_speed + SPEED_INCREMENT, MAX_BALL_SPEED)
                if score2 == WINNING_SCORE:
                    game_over = True
                else:
                    ball_dx, ball_dy = reset_ball(ball, ball_speed)

            if ball.right >= WIDTH:
                score1 += 1
                ball_speed = min(ball_speed + SPEED_INCREMENT, MAX_BALL_SPEED)
                if score1 == WINNING_SCORE:
                    game_over = True
                else:
                    ball_dx, ball_dy = reset_ball(ball, ball_speed)

        screen.fill(BACKGROUND_GRAY)
        pygame.draw.rect(screen, PADDLE_BLUE, paddle1)
        pygame.draw.rect(screen, PADDLE_BLUE, paddle2)
        pygame.draw.ellipse(screen, BALL_ORANGE, ball)
        draw_scores(screen, score_font, score1, score2)

        if game_over:
            text = game_over_font.render("Game Over! Press ENTER to Restart", True, (200, 200, 200))
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()
        clock.tick(FPS)


def main():
    cap = None
    try:
        pygame.init()
        cap = open_camera()
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                               min_tracking_confidence=0.5)
        while run_game(hands, cap):
            pass  # run_game returns True to play again
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)
    finally:
        if cap is not None:
            cap.release()
        pygame.quit()


if __name__ == "__main__":
    main()
