import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import random 
import numpy as np

# Datenstrucktur in die das Memory gespeichert wir
from collections import deque

import gym
import gym_TSP
import pygame

import matplotlib
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output
from itertools import count

plt.ion() # macht den Plot interaktiv

from tqdm import tqdm

env = gym.make("TSPEnv-v0", render_mode="rgb_array", size=10)
observation, info = env.reset()

# set up pygame
pygame.init()

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LearningRate = 0.001

class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        
        self.linear1 = nn.Linear(inputSize,hiddenSize) # input Layer
        self.linear2 = nn.Linear(hiddenSize,outputSize) # hidden Layer
        self.linear3 = nn.Linear(outputSize,outputSize) # output Layer
    
    def forward(self, x): # ist die "prediction" Funktion
        x = F.relu(self.linear1(x)) # Aktivierungsfunktion
        x = self.linear2(x)
        
        return x
    
    def save(self, fileName='model.pth'): # dient dazu das Model zu speichern
        modelFolderPath = './model'
        
        if not os.path.exsits(modelFolderPath):     # wenn es den Ordner noch nicht gibt wird hier ein neuer erzeugt
            os.makedirs(modelFolderPath)
            
        fileName = os.path.join(modelFolderPath, fileName) # hier wird der gesammte Filename zusammengesetzt
        torch.save(self.state_dict(), fileName) # das Model speicher 
        
class QTrainer:
    def __init__(self, model, learningRate, gamma):
        self.model = model
        self.learningRate = learningRate
        self.gamma = gamma
        
        self.optimizer = optim.Adam(model.parameters(), lr = self.learningRate) # als Otimizer Adam gewählt kann aber auch asugewechselt werden
        self.criterion = nn.MSELoss() # loss Funktion mit "Mean Squared Error"
        
    def train_step(self, state, action, reward, nextState, done):
        state = torch.tensor(state, dtype = torch.float)
        nextState = torch.tensor(nextState, dtype = torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch.float)
        
        # die volgenden Schritte dienen dazu das sowohl einzelne werte als auch Batches von werten verarbeitet werden können        
        if len(state.shape) == 1:
            # die Werte werden in der Form (n,x) benötig weshalb hier noch eine dimension hinzugefüght werden muss
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # touple mit nur einem wert
            
        # Schritt 1: predictet Q-Werte mit dem actuellen Wert
        prediction = self.model(state)
        
        # Schritt 2: Formel reward + gamme * max(nextPredictetQValue) nur wenn done = False
        tmp = prediction.clone() 
        
        for index in range(len(done)):
            QNew = reward[index]
            if not done[index]:
                QNex = reward[index] + self.gamma * torch.max(self.model(nextState[index])) # die Oben genannte Formel wird hier angewant
                
            tmp[index][torch.argmax(action).item()] = QNew
            
        # Schritt 3: Loss Funktion    
        self.optimizer.zero_grad() # Funktion um die Gradient zu leeren
        loss = self.criterion(tmp, prediction) # repräsentieren QNew and Q
        loss.backward() # backpropagation anwenden und gradient setzen
        
        self.optimizer.step()

def getState():
    targetsOpen = env.get_targets_target_location_open()
    targetsDone = env.get_target_location_done()
    agentPosition = env.get_agentPosition()
    
    agentRow, agentColum = agentPosition
    state = []
    
    for i in range(0, len(targetsOpen)):
        targetRow, targetColum = targetsOpen[i]
        state.append(((targetRow-agentRow)**2)+((targetColum-agentColum)**2)) # die oben gennante Funktion
        
    # damit der State immer die Selbe größe hat werden die geschlossenen Ziele mit 0 aufgefüllt    
    for i in range(0, len(targetsDone)):
        state.append(0)
        
    """für das Testen der Funktion während des Betriebs"""
    #print(state)
    
    return state


class Agent:
    
    def __init__(self):
        # um Nachzuvolziehen wie viele durchläufe es gab
        self.NumberOfFoundTargets = 0
        # steuert den übergang zwichen random actions und gelernten actions
        self.epsilon = 0 
        # discount rate muss kleiner als Eins sein uns ist meist 0.8 oder 0.9
        self.gamma = 0.9
        # das "Gedächtnis" wenn das deque "voll" ist wird ein popleft() ausgeführt
        self.memory = deque(maxlen = MAX_MEMORY)
        
        # inputSize entspricht der größe des States welcher wiederum der anzahl der Ziele entspricht
        inputSize = len(env.get_targets_target_location_open())
        # hiddenSize als konstante ausgesucht dieser wert kann angepasst werden
        hiddenSize = 256 
        # outputSize entpricht der anzahl der Möglichen actions hier 4
        outputSize = 4
        
        self.model = LinearQNet(inputSize, hiddenSize, outputSize) 
        self.trainer = QTrainer(self.model, learningRate=LearningRate, gamma=self.gamma)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((self, state, action, reward, next_state, done)) # wird als Touple angehängt
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE)
        else:
            miniSample = self.memory 
        
        states, actions, rewards, next_states, dones = zip(*miniSample) # zip sorgt dafür das die Daten richtig extrahiert werden damit alle rewards zusammen sind
        self.trainer.train_step(states, actions, rewards, next_states, dones)           
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        # das epsison steuert das verhältniss zwichen exploration / explotatio
        # exploration wird durch random moves gesteuert damit der Agent das Enviornment erkundet
        # explotation ist das anwenden des gesammelten Wissen
        
        self.epsilon = 80 - self.NumberOfFoundTargets
        
        # die Art wie eine action ausgewählt wird weicht hier vom Tutorial ab da dort eine andere steuerungslogik verwendet wird
        
        # ließt die möglichen actions aus dem Enviorment aus
        n_actions = env.action_space.n
        
        if random.randint(0,200) < self.epsilon: # daher das das epsilon negativ werden kann, wenden hier irgendwann keine random actions mehr gewählt
            action = random.randrange(n_actions) # auswahl einer Zufälligen action
        else: 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            action = torch.argmax(prediction).item() # hier müssen vielleicht noch änderungen gemacht werden 
            
        return action

def plot(FinalRewards, MeanRewards):
    display.clear_output(wait = True)
    display.display(plt.gcf()) # um in die Aktuelle Figure zu Plotten wenn es eine gibt sonst wird eine erzeugt
    plt.clf() # clear die Aktuelle Figur
    plt.titel('TSP Training')
    plt.xlabel('Number of episodes')
    plt.ylabel('Reward')
    plt.plot(FinalRewards)
    plt.plot(MeanRewards)
    plt.ylim(ymin = 0) # setzt das Limit für die Y-Werte
    plt.text(len(FinalRewards)-1, FinalRewards[-1], str(FinalRewards[-1]))
    plt.text(len(MeanRewards)-1, MeanRewards[-1], str(MeanRewards[-1]))

def train(render = False):
    # um die ergebnisse der einzelnen Durchgänge zu speichen und am ende auszugeben
    plot_FinalRewards = []
    plot_MeanRewards = []
    plot_maxReward = 0
    maxReward = 0
    
    # um die Episoden anzahl zu setzen 
    numEpisodes = 3

    agent = Agent()
    
    for i in tqdm(range(numEpisodes)):
        # das Enviornment für die Nächste Episode zurücksetzten (hier kann auch der Parameter Size mit gegeben werden)
        env.reset()
        
        # terminated ist true wenn das TSP Problem gelöst wurde
        terminated = False
        
        # zu ermitteln ob ein neuer Punkt abgearbeitet wurde
        done = False
        doneTargets = env.get_target_location_done()
        
        while not terminated:
            print("Steps:")
            print(env.get_Steps())
            
            print("Offene Ziele:")
            print(env.get_openTargets())
            
            print("Fertige Ziele:")
            print(env.get_closedTargets())
            
            #get old State
            print("Old State:")
            stateOld = getState()

            # action treffen
            action = agent.get_action(stateOld)

            # action ausführen
            _, reward, terminated, _, _ = env.step(action)

            #get new State
            print("New State:")
            stateNew = getState()
            
            # für die ausgabe des Rewasrd
            print("Reward:")
            print(reward)
            
            if doneTargets != env.get_target_location_done():
                doneTargts = env.get_target_location_done()
                done = True

            # das short_momory trainieren
            agent.train_short_memory(stateOld, action, reward, stateNew, done)

            # remeremember 
            agent.remember(stateOld, action, reward, stateNew, done)

            if render: # Hinzugefüght damit das rendering optional gesteuert werden kann
                # Ausgabe von dem was der Agent grade macht 
                clear_output(wait=True)
                plt.figure(figsize=(10.,24.))
                plt.imshow(env.render())
                #plt.title(f'Step: {t}, Reward: {rewardEnv}')
                #targets_open = env.get_targets_target_location_open()
                #for i in range(0, len(targets_open)):
                #    x, y = targets_open[i]
                #    plt.text(x//3, y//3, f'({x//9},{y//9})')
                plt.show()

            if done:
                # das long_memory trainieren 
                observation, info = env.reset()

                agent.NumberOfFoundTargets += 1

                # das lang Zeit memorytrainieren
                agent.train_long_memory()

                # maximalen reward setzen
                if reward > maxReward:
                    plot_maxReward = reward
                    agent.model.save()

                print('Episode: ', agent.NumberOfEpisode, 'Reward: ', reward, 'Derzeitiger MaxReward: ', plot_maxReward)

                plot_FinalRewards.append(reward)
                totalReward += reward
                plot_MeanRewards.append(totalReward / agent.NumberOfFoundTargets)
                plot(plot_FinalRewards, plot_MeanRewards)
                
                done = False

print("Start Training...")
train(render = True)