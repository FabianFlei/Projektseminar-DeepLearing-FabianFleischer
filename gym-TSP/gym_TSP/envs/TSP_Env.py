#Mehere Ziele abgeleitet von https://github.com/TheoLvs/reinforcement-learning/blob/master/5.%20Delivery%20Optimization/delivery.py

#gym-TSP\gym_TSP\envs\TSP_Env.py

import gym # für das Enviorment zuständig
from gym import spaces

import pygame # für die Visualisierung zuständig und auführung der Actionen
import numpy as np
from numpy import random
from numpy.linalg import norm
from DataExtractor import dataExtractor

# Datenstrucktur in die das Memory gespeichert wir
from collections import deque

# wird zur berechnung der euklidichen distanz benötigt
import mpmath 

class TSPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24,}
    
    def __init__(self, render_mode = None, dataSet = None, size = None, maxMemory = 100):
        # initalisierung Ausgabe Variablen
        self.steps = 0 # zur Ausgabe: wieviele Steps wurden gemacht
        self.openTargets = 0 # zur Ausgabe: wieviele Targets sind noch offen
        self.closedTargets = 0 # zur Ausgabe: wieviele Targets sind schon abgearbeitet

        # initalisierung Env Variablen
        self.size = size
        self.dataSet = dataSet
        self.window_size = 800 #size of the PyGame window
        self.dataSetOrSize = True if self.dataSet != None else False

        # initalisierung Mermory für gemachte schritte
        self.memory = self.memory = deque(maxlen = maxMemory)

        # spezielle initalisierungen wenn ein Datensatz verwendet wird
        if self.dataSetOrSize:
            _, info = dataExtractor.extractData(dataSet)

            self.size = info["maxValue"]//10 #size of the square grid
            self.maxValue = info["maxValue"]//10
            self.minValue = info["minValue"]//10
            self.offset = info["offset"]
            self.dimension = info["dimension"]  # -1 wenn der 52. Punkt nicht richtig angezeigt wird
            self.quotient = 1
            self.oneOrZero = 1

            #Observations sind dictionaries mit der Positions des agent und des Ziels
            # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, self.size-1, shape=(2,), dtype=int),
                }
            )

            self.openTargets = self.dimension # setzt initial die Anzahl der noch offenen Ziele

        else:
            self.quotient = 1
            self.oneOrZero = 0

            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, self.size, shape=(2,), dtype=int ),
                }
            )   

            self.openTargets = self.size # setzt initial die Anzahl der noch offenen Ziele



        #o = self.observation_space["targets_open"].sample()

        # Weil wir vier Bewegungsmöglichkeiten haben ist die action_space = 4
        self.action_space = spaces.Discrete(4)
        
        #der Agent kann vier actionen durchführen "right", "up", "left", "down"
        #hier werden actionen in richtungen umgewandelt
        self._action_to_direction = {
            0: np.array([1,0]), # nach rechts bewegen
            1: np.array([0,1]), # nach oben bewegen
            2: np.array([-1,0]), #nach links bewegen
            3: np.array([0,-1]), #nach unten bewegen
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def get_size(self):
        return self.size

    def get_Steps(self):
        return self.steps

    def get_openTargets(self):
        return self.openTargets

    def get_closedTargets(self):
        return self.closedTargets

    def get_Terminated(self):
        return self.terminated

    def get_agentPosition(self):
        return self._agent_location

    def get_targets_target_location_open(self):
        return self._target_location_open

    def get_target_location_done(self):
        return self._target_location_done

    def get_printInfo(self):
        return self.printInfo

    def get_Reward(self):
        return self.reward

    def showRoute(self):
        route = []
        route = np.array(self.startPoint)
        route = list(route)
        for elemts in self._target_location_done:
            route.append(np.array(elemts))
        return route
        
    def _get_obs(self):
        return {"agent": self._agent_location}

    def _get_info(self):
        # hier könnte man self._target_location_done zurükgeben
        #funktioniert aber so nicht 
        #return {np.array(self._target_location_done)}
        return {"reward": self.reward}
    # wenn wir auf informationen die nur in der step methode sind zugreifen wollen müssen wir das dictionary das bei _get_info in step zurückgegeben ist updaten

    def _getState(self):
        # diese Methode dient ermöglicht es den Zielen eine affection zu geben und auch einen Nagativen reward für das laufen in die Außenwand zu geben
        
        agentRow, agentColum = self._agent_location
        state = []
        
        for i in range(0, len(self._target_location_open)):
            targetRow, targetColum = self._target_location_open[i]
            state.append(((targetRow-agentRow)**2)+((targetColum-agentColum)**2)) # die oben gennante Funktion
            
        # damit der State immer die Selbe größe hat werden die geschlossenen Ziele mit 0 aufgefüllt    
        for i in range(0, len(self._target_location_done)):
            state.append(0)
        
        return state

    def _get_euklidische_Distanz(self):
        distance = 0

        for i in range(0, len(self._target_location_done)):
            xi,yi = self._target_location_done[i]

            if i+1 < len(self._target_location_done):
                xj,yj = self._target_location_done[i+1]
            else:
                xj,yj = self._target_location_done[-1] # für den Letzen Punkt wird der weg zum Startpunkt verbunden damit ein rundweg entsteht

            x = xi - xj
            y = yi - yj

            distance += mpmath.nint(mpmath.sqrt(x * x + y * y))

        return distance

    # reset initialieiert eine neue episode und wir immer aufgerufen wenn step done zurückgibt
    def reset(self, seed=None, options=None, size=None, **hyperparam):

        # Ausgabewerte zurücksetzen
        # zur Ausgabe: wieviele Steps wurden gemacht
        self.steps = 0 

        # zur Ausgabe: wieviele Targets sind schon abgearbeitet
        self.closedTargets = 0 

        # zur Ausgabe: wieviel punkte mehrfach besucht worden
        self.visitedTwice = 0

        # hier geben wir self.np_random einen seed 
        super().reset(seed=seed)

        # für die variable erhöhung der size während des Trainings (nur mit convolutional neural network)
        if size != None:
            self.size = size

        # liste für die Fertigen Ziele 
        self._target_location_done = []

        if self.dataSetOrSize:
            # Ziele ertselle unter der Berücksichtigung des Ofsets
            self._target_location_open, info = dataExtractor.extractData(self.dataSet)
            self._target_location_open = self._target_location_open
            
            """Der letzte Punkt wird momentan gelöscht da dieser ausseralb des Sichbaren und begehbaren raums liegt"""
            del self._target_location_open[-1]
        else:
            self._target_location_open = []

            
            while len(self._target_location_open) < self.size:
                x, y = self.np_random.integers(0, self.size, size=2, dtype=int)
                
                NoDuplicate = True

                # hier wird überprüft das nicht zufällig zwei Ziele an der selben Koordinate erstellt werden
                for i in range(len(self._target_location_open)):
                    if np.array_equal(self._target_location_open[i], np.array([x,y])):
                        NoDuplicate = False 
                        break                    
            
                if NoDuplicate:
                    self._target_location_open.append(np.array([x,y]))

        
        # eine zufällige position für den Agent auswählen 
        self.indexStartPoint = random.randint(0, len(self._target_location_open)-self.oneOrZero)
        self._agent_location = self._target_location_open[self.indexStartPoint]//self.quotient # für random Agenent start location self.np_random.integers(0, self.size, size = 2, dtype = int))
        
        #start punkt setzen
        self.startPoint = self._agent_location

        # reward initialisieren 
        self.reward = 0

        # das Eepsilon bestimmt für wie viele Schritte die Targets eine Affection haben
        self.epsilon = 6000

        # zur Ausgabe: wieviele Targets sind noch offen
        self.openTargets = self.dimension if self.dataSetOrSize else len(self._target_location_open) 
            
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()


        return observation, info
    
    def step(self, action):
        oldState = self._getState()

        # initialisierung des stepRewards
        stepReward = 0 

        # der indexTargetAffection bestimmt von welchem Target die Affection ausgeht
        tmpListe = oldState.copy()

        # alle Elemente aus der Liste die 0 sind müssen entfernt werden da diese Punkte
        # bereits abgearbeitet sind und keine affection mehr haben sollen       
        tmp = min(tmpListe)
        
        while tmp == 0 and len(tmpListe) != 1: 
            tmpListe.remove(0)          
            tmp = min(tmpListe)

        indexTargetAffection = oldState.index(tmp)

        # überprüfung das der Startpunkt keine Affection bekommt
        if indexTargetAffection == self.indexStartPoint and (len(tmpListe) != 1): 
            tmpListe.pop(self.indexStartPoint)
            tmp = min(tmpListe)
            indexTargetAffection = oldState.index(tmp)

        tmpOldState = tmp
        tmpListeOldState = tmpListe
        indexTargetAffectionOldState = indexTargetAffection

        # für die Ausgabe der gemachten Schritte
        self.steps += 1

        # Mapping der action auf die richtig in die gelaufen werden soll
        direction = self._action_to_direction[action]
        
        # hier wird np.clip genutzt damit der agent nicht das grid verlassen kann
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size -1)

        # reward für den algorithmus 1 für einen ereichten punkt -0.1 für jede action bis zum punkt und -1 wenn ein bereits abgearbeiteter punkt erneut besucht wird
        
        # negativer reward für jeden gemachten step
        stepReward -= 10

        afterTargetDoneIndexTargetAffection = indexTargetAffection        

        # Ziele von der Open Liste nehmen und als done makieren + prositiver reward für abgearbeiteten Punkt
        for i in range(0,len(self._target_location_open)):
            if np.array_equal(np.array(self._agent_location), np.array(self._target_location_open[i]//self.quotient)):
                if np.array_equal(np.array(self._agent_location), np.array(self.startPoint)):
                    # dieser Teil ist dafür da das der Letzte Punkt wieder der Startpunkt sein muss
                    if len(self._target_location_open)-self.oneOrZero == 1:
                        self._target_location_done.append(self._target_location_open[i])
                        stepReward += 100000 # reward für das Beenden der Route
                        del self._target_location_open[i]

                        # für die Ausgabe Offene/Fertige Ziele
                        self.openTargets -= 1
                        self.closedTargets += 1

                        break
                    else:
                        break 
                
                self._target_location_done.append(self._target_location_open[i])
                stepReward += 100000
                del self._target_location_open[i]

                # wenn ein Target mit eienem kleinerem Index aus der Liste entfernt wird muss der self.indexStartPoint 
                # dem enstrechend um einen nach Links verschoben werden damit die berechnung der affection weiterhin
                # wie erwartet funktioniert
                if i < self.indexStartPoint:
                    self.indexStartPoint -= 1

                # wenn ein Target mit eienem kleinerem Index aus der Liste entfernt wird muss afterTargetDoneIndexTargetAffection
                # dem enstrechend um einen nach Links verschoben werden damit die berechnung der affection weiterhin
                # wie erwartet funktioniert
                if i < indexTargetAffection:
                    afterTargetDoneIndexTargetAffection -= 1

                # für die Ausgabe Offene/Fertige Ziele
                self.openTargets -= 1
                self.closedTargets += 1
                break
                
        # eine Episode ist fertig wenn der argent alle targets erreicht hat
        self.terminated = True if len(self._target_location_open)-self.oneOrZero == 0 else False
        
        newState = self._getState()

        runAgainstWall = False
        # negativer reward für das Laufen gegen die grenze der Welt
        if newState == oldState:
            stepReward -= 1000000
            runAgainstWall = True

        if indexTargetAffection < 0 or indexTargetAffection >= len(oldState) or indexTargetAffection >= len(newState):
            print(indexTargetAffection)
            print(len(newState))
            print(len(oldState))
            print(self.indexStartPoint)

        # reward wenn der agent in die richtung der Ziele Läuft
        affectionReward = oldState[indexTargetAffection] - newState[afterTargetDoneIndexTargetAffection] # dieser reward kann auch negativ werden
        
        # der affectionReward wird mit den gemachten Schritten immer weiter abflachen
        affectionReward = affectionReward * (self.epsilon/2) if affectionReward < 1 else affectionReward * (self.epsilon/2)
        
        stepReward += affectionReward

        # wenn der Agent an einem ort schon war soll er einen negativen reward erhalten
        for element in self.memory:
            if np.array_equal(self._agent_location, element):
                stepReward -= affectionReward
                stepReward -= 100000

                self.visitedTwice += 1

        # besuchten punkt in das memory setzten
        self.memory.append(self._agent_location)


        self.reward += stepReward


        self.printInfo = {
                        "runAgainstWall":  runAgainstWall,
                        "oldState": oldState,
                        "newState": newState,
                        "indexTargetAffectionOldState": indexTargetAffectionOldState,
                        "affectionReward": affectionReward,
                        "indexStartPoint": self.indexStartPoint,
                        "tmpOldState": tmpOldState,
                        "tmpListeOldState": tmpListeOldState,
                        "stepReward": stepReward,
                        "euklidische_Distanz": self._get_euklidische_Distanz(), 
                        "visitedTwice": self.visitedTwice,
                    }


        # das epsilon decrementieren damit die affection der Punkte nachlässt
        self.epsilon -= 1

        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

            
        return observation, stepReward, self.terminated, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init() # initilation PyGame
            pygame.display.init() # initilation display
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock() # wird für die FPS benötigt 
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))
        
        pix_square_size = (self.window_size / self.size) # für die größe eines quadretes im raster
        

        coordinatesTargetsOpen = []
        # die noch offenen Ziele auf den PyGame Window ausgeben
        for i in range(0, len(self._target_location_open)):
            a =np.array(self._target_location_open[i]//self.quotient)
            pygame.draw.rect(
            canvas,
            (0, 145, 0),
            pygame.Rect(
                pix_square_size * a,
                (pix_square_size, pix_square_size),
            ),
            )
            coordinatesTargetsOpen.append(a)


        # testet ob zwei oder mehrere Punkte die selben Koordinaten haben
        # dies könnte durch die Umrechnung der Daten auftreten bei zwei punkten die sehr nah bei einander sind
        Mehfrach = False
        offsetA = 1
        offsetB = 0
        for a in range(0,len(coordinatesTargetsOpen)):
            for b in range(offsetA,len(coordinatesTargetsOpen)):
                if np.array_equal(coordinatesTargetsOpen[a], coordinatesTargetsOpen[b]): Mehfrach = True 
            for k in range(0, offsetB):
                if np.array_equal(coordinatesTargetsOpen[a], coordinatesTargetsOpen[k]): Mehfrach = True
            offsetA += 1
            offsetB += 1
        self._target_dupplicate = Mehfrach # eingebaut für Fehleranalyse

  
        # den Startpunkt farblich hervorheben
        pygame.draw.rect(
            canvas,
            (255, 192, 0),
            pygame.Rect(
                pix_square_size * self.startPoint,
                (pix_square_size, pix_square_size),
            ),
            )


        # die fertigen Ziele auf den PyGame Window ausgeben
        for i in range(0, len(self._target_location_done)):
            a =np.array(self._target_location_done[i]//self.quotient)
            pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * a,
                (pix_square_size, pix_square_size),
            ),
            )
               
        # den Agent auf dem PyGame Window ausgeben
        pygame.draw.circle(
            canvas,
            (0,0,255),
            (self._agent_location + 0.5)* pix_square_size,
            pix_square_size/2,
        )


        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect()) # kopiert die zufor diefienierten ausgaben in das sichbare Window
            pygame.event.pump()
            pygame.display.update()
            
            self.clock.tick(self.metadata["render_fps"]) # hält die FPS stabiel und setzt die FPS auf den wert in den Metadaten
        else: # the case of rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2))
        
    def close(self): # nachdem man diese methode aufgerufen hat sollte man nicht mehr mit dem enviorment interagieren
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()