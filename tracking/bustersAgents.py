# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"LAURA NEFF"
"laura.neff@emory.edu/lneff2/2304477"
"THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING "
"ANY# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. LAURA NEFF"

import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False): pass
    def update(self, state):    pass
    def pause(self):    pass
    def draw(self, state):    pass
    def updateDistributions(self, dist):    pass
    def finish(self):    pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0: allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent:
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules: inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        for index, inf in enumerate(self.inferenceModules):
            if not self.firstMove and self.elapseTimeEnable:
                inf.elapseTime(gameState)
            self.firstMove = False
            if self.observeEnable:
                inf.observeState(gameState)
            self.ghostBeliefs[index] = inf.getBeliefDistribution()
        self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions

class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that
        has not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (in maze distance!).

        To find the maze distance between any two positions, use:
        self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
        successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief distributions
        for each of the ghosts that are still alive.  It is defined based
        on (these are implementation details about which you need not be
        concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.

        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = [beliefs for i,beliefs
                                            in enumerate(self.ghostBeliefs)
                                            if livingGhosts[i+1]]




        ghostPosition = list() #Make a list for the ghost positions
        for belief in livingGhostPositionDistributions: #For each living ghost's belief distribution
            firstSpaceVal = 0 #Set the probability of a space in the belief distribution to 0 for when we calculate a minimum later
            mostLikelyPosition = None #Set the most likely position for the ghost to be to null for when we calculate a minimum later
            for k, v in belief.items(): #For each key, value pair in each ghost's belief distribution
                if v > firstSpaceVal: #If the probability of the ghost being in one space in the ghost's belief distribution is greater than the one we saw before
                    mostLikelyPosition = k #Then now, we know where the ghost is more likely to be coordinate-wise; take the position associated with probability
                    firstSpaceVal = v #Store the more likely probability of where the ghost would be
            ghostPosition.append(mostLikelyPosition) #Store the most-likely position of where EACH ghost would be in the ghostPositions list

        minimum = 999999 #Initialize minimum to a big number for comparisons later for when we calculate the real minimum
        closestGhost = None #Set the closest ghost to null because we don't know what's the closest ghost yet
        for position in ghostPosition: #For each most-likely position of each ghost we have
            ghostDistance = self.distancer.getDistance(position, pacmanPosition) #Calculate the maze distance between a ghost (where it will likely be) and Pacman
            if ghostDistance < minimum: #If the distance is less than the shortest distance already calculated between Pacman and another ghost
                minimum = ghostDistance #Set the minimum to the new shortest distance
                closestGhost = position #Store the position associated with the ghost that is closest to Pacman

        bestDistance = 999999 #Initialize the best distance to a big number for comparisons later when we calculate a minimum
        bestAction = None #Set the best action Pacman can take to null
        pacmanDistance = self.distancer.getDistance(closestGhost, pacmanPosition) #Calculate the maze distance between Pacman and the closest ghost to him
        for action in legal: #for each legal action
            successorPosition = Actions.getSuccessor(pacmanPosition, action) #Find a successor position of Pacman
            successorDistance = self.distancer.getDistance(closestGhost, successorPosition) #Calculate the distance between that successor position and the closest ghost
            if successorDistance < bestDistance: #If the successor action brought Pacman closer to the ghost than he was before
                bestAction = action #Then we know that's the best action Pacman can take to hunt the ghost
                bestDistance = successorDistance #Then we know that action will help us find the shortest path to the ghost
        return bestAction



