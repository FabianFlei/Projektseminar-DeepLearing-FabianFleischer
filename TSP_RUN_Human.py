
import gym
import gym_TSP
import pygame

env = gym.make("TSPEnv-v0", render_mode="human", dataSet = "berlin52.tsp")

pygame.init()

observation, info = env.reset()
run = True
state = env.get_target_location_open()

while run:
    get_event = pygame.event.get()
    
    for event in get_event:
        
        if event.type == pygame.QUIT:
            env.close()
            run = False

        # hier beginnt der Eigenanteil
        if event.type == pygame.KEYDOWN: 

            if event.key == pygame.K_DOWN: # nach unten bewegen
                observation, reward, terminated, boo, info = env.step(action = 1)
                if terminated:
                    env.close()
                    run = False

            if  event.key == pygame.K_UP: # nach oben bewegen
                observation, reward, terminated, boo, info = env.step(action = 3)
                if terminated:
                    env.close()
                    run = False

            # tourning left and right     
            if event.key == pygame.K_RIGHT: # nach rechts bewegen
                observation, reward, terminated, boo, info = env.step(action = 0)
                if terminated:
                    env.close()
                    run = False
                
            if event.key == pygame.K_LEFT: # nach links bewegen
                observation, reward, terminated, boo, info = env.step(action = 2)
                if terminated:
                    env.close()
                    run = False 
                    
        env.render()
            
#Tasten Beschreibung  https://www.pygame.org/docs/ref/key.html#key-constants-label