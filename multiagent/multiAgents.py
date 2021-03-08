# multiAgents.py
# --------------
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
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util, copy, math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    @staticmethod
    def distance(num1, num2):
        distance = math.sqrt(((num1[0] - num2[0]) ** 2) + ((num1[1] - num2[1]) ** 2))
        return distance;

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #power pellet capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        ghostPositions = successorGameState.getGhostPositions()

        scaredGhostWeight = .5 #will get better score by a lot
        capsuleWeight = .3 #for opportunity for better score
        foodWeight = .2 #will increase score, but not by much

        oh_no_a_ghost = .1

        pacman_guinea_pig = successorGameState # copy.copy(currentGameState)

        calcScore1 = 0
        calcScore2 = 0


        #if(newScaredTimes > 0 and one of the successor positions is a ghost state):
            #change Pacman guinea pig's position to that and calculate score

        for ghost in newGhostStates:
            dist_to = ReflexAgent.distance(newPos, ghost.configuration.pos)
            if ghost.scaredTimer > 0:
                if dist_to == 0:
                    pacman_guinea_pig.pacmanPosition = ghost.configuration.pos
                    calcScore1 = pacman_guinea_pig.getScore() * scaredGhostWeight
                    print(calcScore1)
            elif dist_to < 3:
                calcScore2 = successorGameState.getScore() * oh_no_a_ghost
                print(calcScore2)

        # for k in newScaredTimes:
        #     if (k > 0):
        #        for j in ghostPositions:
        #             if(ReflexAgent.distance(newPos,j) == 0):
        #                 pacman_guinea_pig.pacmanPosition = j
        #                 calcScore1 = pacman_guinea_pig.getScore() * scaredGhostWeight
        #                 print(calcScore1)
        #
        # for k in newScaredTimes:
        #     if (k == 0):
        #         for j in ghostPositions:
        #                 #print(i)
        #                 #print(ghostPositions)
        #             if(manhattanDistance(newPos, j) < 3):
        #                 #newPos.remove(i) Can we do this according to the algorithm of eval functions?
        #                 calcScore2 = successorGameState.getScore() * oh_no_a_ghost
        #                 print(calcScore2)

        if newFood[newPos[0]][newPos[1]] is True:
            calcScore3 = successorGameState.getScore() * foodWeight
            #print(calcScore3)


        #but if any of these scores is less than normal score, go with normal score








        #print(currentGameState.getPacmanPosition())












        #other factors to consider:
            #distance from alive ghost
            #if alive ghost is in the successor state, turn away
        #if from the successor state, there's a lot of food, go in that direction maybe than other








        #print("successorGameState: ", successorGameState.getPacmanPosition())
        #print("ghost position: ", ghostPositions)
        #print("newPos: ", newPos)
        # print("currentFood: ", currentFood)
        # print("newFood: " , newFood)
        # print("currentCapsules ", currentCapsules)
        # print("newCapsules: " , newCapsules)
        # print("newGhostStates: " , newGhostStates)
        # print("newScaredTimes: ", newScaredTimes)

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

