import pygame
import random
import cv2
import mediapipe as mp
import time

pygame.init()

WIDTH, HEIGHT = 1550, 800  
BALL_SIZE = 30
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 120
INITIAL_BALL_SPEED = 10     
PADDLE_SPEED = 7
FPS = 60
PAUSE_DURATION = 2
MAX_BALL_SPEED = 15        
SPEED_INCREMENT = 2         

BALL_ORANGE = pygame.Color("#F97C00")
PADDLE_BLUE = pygame.Color('#E2DFD0')
BACKGROUND_GRAY = pygame.Color('#0C0C0C')
SCORE_GOLD = (255, 215, 0)

# Player names
PLAYER1_NAME = 'A'
PLAYER2_NAME = 'B'

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ping Pong")

# Mediapipe setup for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Timer for ball pause after scoring
ball_paused = False
pause_start_time = 0


# Function to draw scores
def draw_scores(score1, score2):
    font = pygame.font.Font(None, 36)
    score1_text = font.render(f"{PLAYER1_NAME} : {score1}", True, SCORE_GOLD)
    score2_text = font.render(f"{PLAYER2_NAME} : {score2}", True, SCORE_GOLD)
    screen.blit(score1_text, (10, 10))
    screen.blit(score2_text, (WIDTH - score2_text.get_width() - 10, 10))


# Reset ball position and direction
def reset_ball(BALL, BALL_SPEED):
    global ball_paused, pause_start_time

    BALL.center = (WIDTH // 2, HEIGHT // 2)
    ball_paused = True
    pause_start_time = time.time()

    ball_dx = BALL_SPEED * random.choice((1, -1))
    ball_dy = BALL_SPEED * random.choice((1, -1))
    return ball_dx, ball_dy


def run_game():
    global ball_paused, pause_start_time

    # Initialize game objects
    BALL = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
    PADDLE1 = pygame.Rect(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
    PADDLE2 = pygame.Rect(WIDTH - 60, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

    BALL_SPEED = INITIAL_BALL_SPEED
    ball_dx, ball_dy = reset_ball(BALL, BALL_SPEED)
    ball_in_motion = False

    score1, score2 = 0, 0
    winning_score = 5

    clock = pygame.time.Clock()
    running = True
    game_over = False

    while running:
        # Capture frame from camera
        success, image = cap.read()
        if success:
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
        else:
            results = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  
            if game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return True   

        if not game_over:
            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_y = hand_landmarks.landmark[8].y * HEIGHT
                    if hand_landmarks.landmark[8].x < 0.5:
                        PADDLE1.y = int(hand_y) - PADDLE_HEIGHT // 2
                    else:
                        PADDLE2.y = int(hand_y) - PADDLE_HEIGHT // 2

            PADDLE1.y = max(0, min(HEIGHT - PADDLE_HEIGHT, PADDLE1.y))
            PADDLE2.y = max(0, min(HEIGHT - PADDLE_HEIGHT, PADDLE2.y))

            if ball_paused and time.time() - pause_start_time >= PAUSE_DURATION:
                ball_paused = False
                ball_in_motion = True

            if ball_in_motion:
                BALL.x += ball_dx
                BALL.y += ball_dy

            if BALL.top <= 0 or BALL.bottom >= HEIGHT:
                ball_dy *= -1

            if BALL.colliderect(PADDLE1) or BALL.colliderect(PADDLE2):
                ball_dx *= -1

            if BALL.left <= 0:
                score2 += 1
                BALL_SPEED = min(BALL_SPEED + SPEED_INCREMENT, MAX_BALL_SPEED)
                if score2 == winning_score:
                    game_over = True
                else:
                    ball_dx, ball_dy = reset_ball(BALL, BALL_SPEED)

            if BALL.right >= WIDTH:
                score1 += 1
                BALL_SPEED = min(BALL_SPEED + SPEED_INCREMENT, MAX_BALL_SPEED)
                if score1 == winning_score:
                    game_over = True
                else:
                    ball_dx, ball_dy = reset_ball(BALL, BALL_SPEED)

        screen.fill(BACKGROUND_GRAY)
        pygame.draw.rect(screen, PADDLE_BLUE, PADDLE1)
        pygame.draw.rect(screen, PADDLE_BLUE, PADDLE2)
        pygame.draw.ellipse(screen, BALL_ORANGE, BALL)
        draw_scores(score1, score2)

        if game_over:
            font = pygame.font.Font(None, 72)
            text = font.render("Game Over! Press ENTER to Restart", True, (200, 200, 200))
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()
        clock.tick(FPS)


while True:
    play_again = run_game()
    if not play_again:
        break

cap.release()
pygame.quit()
