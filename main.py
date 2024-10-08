import pygame
import random
import math
import pandas as pd
import torch
from Neuro import model

data = []

def record_data(state, action):
    data.append([*state, action])

def save_data():
    df = pd.DataFrame(data, columns=['circle_x', 'circle_y', 'finish_x', 'finish_y', 'action'])
    df.to_csv('game_data.csv', index=False)


model.eval()
def choose_action(state):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32)
        q_values = model(state)
        action = torch.argmax(q_values).item()
    return action


pygame.init()
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))

blue = (0, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)

circle_radius = 30
circle_speed = 8
circle_color = blue
circle_pos = [screen_width // 2, screen_height // 2]
circle_landed = False

finishRad = 100
finishColor = white
finishPos = [random.randrange(640), random.randrange(480)]
finished = False
efficiency = 0

action_map = {
    (1, 0, 0, 0): 0,  # Влево
    (0, 1, 0, 0): 1,  # Вправо
    (0, 0, 1, 0): 2,  # Вверх
    (0, 0, 0, 1): 3   # Вниз
}


start_time = pygame.time.get_ticks()
time_since_reset = 0

def check_finished(circle_pos, finish_pos):
    return (abs(circle_pos[1] - finish_pos[1]) <= 60) and (abs(circle_pos[0] - finish_pos[0]) <= 60)

def calculate_distance(pos1, pos2):
    return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # save_data()
            pygame.quit()
            quit()

    if not circle_landed and not finished:
        #Human
        # keys = pygame.key.get_pressed()
        # action = (keys[pygame.K_LEFT], keys[pygame.K_RIGHT], keys[pygame.K_UP], keys[pygame.K_DOWN])
        # if action in action_map:
        #     record_data(circle_pos + finishPos, action_map[action])
        # if keys[pygame.K_LEFT]:
        #     circle_pos[0] -= circle_speed
        # if keys[pygame.K_RIGHT]:
        #     circle_pos[0] += circle_speed
        # if keys[pygame.K_UP]:
        #     circle_pos[1] -= circle_speed
        # if keys[pygame.K_DOWN]:
        #     circle_pos[1] += circle_speed
        # #Neuro
        state = circle_pos + finishPos
        action = choose_action(state)
        
        if action == 0:  # Влево
            circle_pos[0] -= circle_speed
        elif action == 1:  # Вправо
            circle_pos[0] += circle_speed
        elif action == 2:  # Вверх
            circle_pos[1] -= circle_speed
        elif action == 3:  # Вниз
            circle_pos[1] += circle_speed

        

        if circle_pos[0] < 0 or circle_pos[1] < 0 or circle_pos[0] > screen_width or circle_pos[1] > screen_height:
            circle_landed = True

        finished = check_finished(circle_pos, finishPos)

    if circle_landed:
        circle_pos = [screen_width // 2, screen_height // 2]
        circle_color = blue
        circle_landed = False

    if finished:
        distance_to_finish = calculate_distance([screen_width // 2, screen_height // 2], finishPos)
        if time_since_reset > 0:  
            efficiency = distance_to_finish / time_since_reset
        else:
            efficiency = 0
        circle_pos = [screen_width // 2, screen_height // 2]
        circle_color = blue
        finished = False
        finishPos = [random.randrange(640), random.randrange(480)]
        start_time = pygame.time.get_ticks()

    if start_time is not None:
        time_since_reset = (pygame.time.get_ticks() - start_time) / 1000 

    screen.fill(black)
    pygame.draw.circle(screen, finishColor, finishPos, finishRad)
    pygame.draw.circle(screen, circle_color, circle_pos, circle_radius)


    font = pygame.font.SysFont(None, 36)
    time_text = font.render(f'Время: {time_since_reset:.2f}s', True, white)
    screen.blit(time_text, (10, 10))


    distance_to_finish = calculate_distance([screen_width // 2, screen_height // 2], finishPos)
    distance_text = font.render(f'Растояние: {distance_to_finish:.2f}', True, white)
    screen.blit(distance_text, (10, 50))


    efficiency_text = font.render(f'Коэф. эффективности: {efficiency:.2f}', True, white)
    screen.blit(efficiency_text, (10, 90))

    pygame.display.update()
    pygame.time.Clock().tick(60)