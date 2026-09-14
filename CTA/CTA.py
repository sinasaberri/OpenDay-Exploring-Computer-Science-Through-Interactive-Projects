import cv2
import pygame
import random
import sys
import time

pygame.init()
WIDTH, HEIGHT = 400, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)

font = pygame.font.Font(None, 42)

fruits = [
    {"color": RED, "score": 1},
    {"color": ORANGE, "score": 2},
    {"color": YELLOW, "score": 3}
]
APPLE_SIZE = 30
APPLE_MIN_SPEED = 3
APPLE_MAX_SPEED = 5
apples = []
basket_fruits = [] 

score = 0
collected = 0  



cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


mouth_x = WIDTH // 2   
mouth_w = 100
mouth_h = 40
last_seen_time = time.time()


def add_apple():
    fruit = random.choice(fruits)
    x = random.randint(WIDTH//4, (WIDTH*3)//4 - APPLE_SIZE)
    y = -APPLE_SIZE
    speed = random.randint(APPLE_MIN_SPEED, APPLE_MAX_SPEED)
    apples.append({
        "x": x, "y": y,
        "speed": speed,
        "color": fruit["color"],
        "score": fruit["score"]
    })


frame_count = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            cap.release()
            pygame.quit()
            sys.exit()

    screen.fill(WHITE)


    frame_count += 1
    if frame_count % 60 == 0:
        add_apple()


    for apple in apples:
        apple["y"] += apple["speed"]
        pygame.draw.circle(screen, apple["color"],
                           (apple["x"] + APPLE_SIZE//2, apple["y"] + APPLE_SIZE//2),
                           APPLE_SIZE//2)


    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face_center = int(x + w/2)
            mouth_x = int(0.85 * mouth_x + 0.15 * face_center)
            last_seen_time = time.time()
            break
    else:
        if time.time() - last_seen_time > 2:
            mouth_x = int(0.9 * mouth_x + 0.1 * (WIDTH//2))


    basket_y = HEIGHT - 60
    basket_rect = (mouth_x - mouth_w//2, basket_y, mouth_w, 50)
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

    text1 = font.render(f"Score: {score}", True, BLACK)
    text2 = font.render(f"Collected: {collected}", True, BLACK)
    screen.blit(text1, (10, 10))
    screen.blit(text2, (10, 50))

    pygame.display.update()
    clock.tick(30)
