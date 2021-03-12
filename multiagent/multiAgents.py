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


"LAURA NEFF"
"laura.neff@emory.edu/lneff2/2304477"
"THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING "
"ANY# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. LAURA NEFF"

from game import Directions
from game import Agent
from game import Actions
from util import manhattanDistance
import random, util, copy, math, time

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
        currentGhostStates = currentGameState.getGhostStates()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ateFood = sum([item for sublist in currentFood.data for item in sublist]) - sum([item for sublist in newFood.data for item in sublist])
        ateCapsule = len(currentCapsules) - len(newCapsules)


        "*** YOUR CODE HERE ***"

        #Evaluation function = w1*f(x) + w2*f(x) + ... wi*f(x)


        ghostPositions = successorGameState.getGhostPositions()

        scaredGhostIsHereWeight = 20 #will get better score by a lot
        scaredGhostNearWeight = 15  #opportunity to get better score
        capsuleAteWeight = 10  #opportunity for better score
        capsuleCloseWeight = 8 + random.random() #opportunity for better score
        foodAteWeight = 3 #will increase score, but not by much
        foodCloseWeight = 1 + random.random()/2



        oh_no_a_ghost = -20 + random.random()

        legalActionCount = successorGameState.getLegalActions()
        legalActionWeight = -15
        calcScore8 = 0
        if legalActionCount < 2:
            calcScore8 = abs(successorGameState.getScore()) * legalActionWeight

        calcScore1 = 0
        calcScore2 = 0
        calcScore3 = 0


        for ghost in currentGhostStates:
            dist_to = ReflexAgent.distance(newPos, ghost.configuration.pos)
            if ghost.scaredTimer > 1:
                calcScore1 += abs(successorGameState.getScore()) * scaredGhostNearWeight * 1/(dist_to+.1)

                if dist_to == 0:
                   calcScore3 += abs(successorGameState.getScore()) * scaredGhostIsHereWeight
            else:
                calcScore2 += abs(successorGameState.getScore()) * oh_no_a_ghost * 1/(min(dist_to,5)+.1)


        foodPositions = set()
        for rowIndex, row in enumerate(currentFood.data):
            for columnIndex, column in enumerate(row):
                if column:
                    foodPositions.add((rowIndex, columnIndex))


        dist_to_food = sum([1/(.1 + ReflexAgent.distance(newPos, i)) for i in foodPositions])

        calcScore4 = abs(successorGameState.getScore()) * foodAteWeight * ateFood

        calcScore5 = abs(successorGameState.getScore()) * foodCloseWeight * dist_to_food



        capsulePositions = set()
        for rowIndex, row in enumerate(currentCapsules):
            for columnIndex, column in enumerate(row):
                if column:
                    foodPositions.add((rowIndex, columnIndex))

        dist_to_capsule = sum([1/(.1 + ReflexAgent.distance(newPos, i)) for i in capsulePositions])

        calcScore6 = abs(successorGameState.getScore()) * capsuleAteWeight * ateCapsule

        calcScore7 = abs(successorGameState.getScore()) * capsuleCloseWeight * dist_to_capsule




        return calcScore1 + calcScore2 + calcScore3 + calcScore4 + calcScore5 + calcScore6 + calcScore7 + calcScore8

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

        def pacValue(gameState, agent, depth):
            # Add terminal condition
            v = []
            #print("minimize other ghosts")
            PacmanActions = gameState.getLegalActions(agent)
            if "Stop" in PacmanActions:
                PacmanActions.remove("Stop")
            if len(PacmanActions) == 0:
                return self.evaluationFunction(gameState)
            for action in PacmanActions:
                #print("pacman moves "+action)
                successor = gameState.generateSuccessor(agent, action)
                newv, _ = value(successor, agent + 1, depth)
                #print("pacman got newv:",newv)
                v.append((newv,action))
                # Returns the successor game state after an agent takes an action
            #print("pacman says:",v)
            return max(v)

        def ghostValue(gameState, agent, depth):
            # Add terminal condition
            v = []
            if agent+1>=gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth - 1
            else:
                nextAgent = agent + 1
                nextDepth = depth
            if len(gameState.getLegalActions(agent)) == 0:
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(agent):
                #print("ghost "+str(agent)+" moves " +action)
                successor = gameState.generateSuccessor(agent, action)
                newv, _ = value(successor, nextAgent, nextDepth)
                #print("ghost got newv:",newv)
                v.append((newv,action))
            #print("ghost says",v)
            return min(v)

        def value(gameState, agent=0, depth=self.depth):
            ##TODO: find terminal state
            #time.sleep(0.05)
            if agent >= gameState.getNumAgents():
                #print("next level")
                return value(gameState, 0, depth - 1)
            #print("agent="+str(agent)+", depth="+str(depth))
            if (depth<1 or gameState.isWin() or gameState.isLose()):
                #print("term state")
                return (self.evaluationFunction(gameState),None)
            if agent==0:
                #print("pacman")
                return pacValue(gameState, agent, depth)
            elif agent < gameState.getNumAgents():
                #print("ghost")
                return ghostValue(gameState, agent, depth)
            else:
                raise Exception("something don goofed.")


        selected = value(gameState)
        return selected[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
            Returns the minimax action with A/B pruning from the current gameState using self.depth
            and self.evaluationFunction.
        """
        "*** YOUR CODE HERE ***"

        def pacValue(gameState, agent, depth, alpha, beta):
            # Add terminal condition
            v = []
            #print("minimize other ghosts")
            PacmanActions = gameState.getLegalActions(agent)
            if "Stop" in PacmanActions:
                PacmanActions.remove("Stop")
            if len(PacmanActions) == 0:
                return self.evaluationFunction(gameState)
            for action in PacmanActions:
                #print("pacman moves "+action)
                successor = gameState.generateSuccessor(agent, action)
                newv, _ = value(successor, agent + 1, depth, alpha, beta)
                #print("pacman got newv:",newv)
                v.append((newv,action))
                highest_v = max(v)
                if highest_v[0] > beta:
                    return highest_v
                alpha = max(alpha,highest_v[0])
                # Returns the successor game state after an agent takes an action
            #print("pacman says:",v)
            return max(v)

        def ghostValue(gameState, agent, depth, alpha, beta):
            # Add terminal condition
            v = []
            if agent+1>=gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth - 1
            else:
                nextAgent = agent + 1
                nextDepth = depth
            if len(gameState.getLegalActions(agent)) == 0:
                return self.evaluationFunction(gameState)
            for action in gameState.getLegalActions(agent):
                #print("ghost "+str(agent)+" moves " +action)
                successor = gameState.generateSuccessor(agent, action)
                newv, _ = value(successor, nextAgent, nextDepth, alpha, beta)
                v.append((newv,action))
                lowest_v = min(v)
                if lowest_v[0] < alpha:
                    return lowest_v
                beta = min(beta, lowest_v[0])
                #print("ghost got newv:",newv)
            #print("ghost says",v)
            return min(v)

        def value(gameState, agent=0, depth=self.depth,alpha=float("-inf"), beta=float("+inf")):
            ##TODO: find terminal state
            #time.sleep(0.05)
            if agent >= gameState.getNumAgents():
                #print("next level")
                return value(gameState, 0, depth - 1)
            #print("agent="+str(agent)+", depth="+str(depth))
            if (depth<1 or gameState.isWin() or gameState.isLose()):
                #print("term state")
                return (self.evaluationFunction(gameState),None)
            if agent==0:
                #print("pacman")
                return pacValue(gameState, agent, depth, alpha, beta)
            elif agent < gameState.getNumAgents():
                #print("ghost")
                return ghostValue(gameState, agent, depth, alpha, beta)
            else:
                raise Exception("something don goofed.")


        selected = value(gameState)
        return selected[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    ##moves = getLegalMoves()
    ##priors = {m:1./len(moves) for m in moves} ???

    #max nodes are as they were in minimax search
    #min nodes are chance nodes (store expected utilities) = children * priors

    def getAction(self, gameState):
        """
          Returns the expectmax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        "*** YOUR CODE HERE ***"

        def pacValue(gameState, agent, depth):
            # Add terminal condition
            v = []
            #print("minimize other ghosts")
            PacmanActions = gameState.getLegalActions(agent)
            if "Stop" in PacmanActions:
                PacmanActions.remove("Stop")
            if len(PacmanActions) == 0:
                return self.evaluationFunction(gameState)
            for action in PacmanActions:
                #print("pacman moves "+action)
                successor = gameState.generateSuccessor(agent, action)
                newv, _ = value(successor, agent + 1, depth)
                #print("pacman got newv:",newv)
                v.append((newv,action))
                # Returns the successor game state after an agent takes an action
            #print("pacman says:",v)
            return max(v)

        def ghostValue(gameState, agent, depth):
            # Add terminal condition
            v = 0
            if agent+1>=gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth - 1
            else:
                nextAgent = agent + 1
                nextDepth = depth
            if len(gameState.getLegalActions(agent)) == 0:
                return self.evaluationFunction(gameState)
            ghostActions = gameState.getLegalActions(agent)
            p = probabilities(gameState, ghostActions)
            for action in ghostActions:
                #print("ghost "+str(agent)+" moves " +action)
                successor = gameState.generateSuccessor(agent, action)
                newv, _ = value(successor, nextAgent, nextDepth)
                #print("ghost got newv:",newv)
                v += p[action]*newv
            #print("ghost says",v)
            return (v,None)

        def value(gameState, agent=0, depth=self.depth):
            ##TODO: find terminal state
            #time.sleep(0.05)
            if agent >= gameState.getNumAgents():
                #print("next level")
                return value(gameState, 0, depth - 1)
            #print("agent="+str(agent)+", depth="+str(depth))
            if (depth<1 or gameState.isWin() or gameState.isLose()):
                #print("term state")
                return (self.evaluationFunction(gameState),None)
            if agent==0:
                #print("pacman")
                return pacValue(gameState, agent, depth)
            elif agent < gameState.getNumAgents():
                #print("ghost")
                return ghostValue(gameState, agent, depth)
            else:
                raise Exception("something don goofed.")
        
        def probabilities(gameState, moves):
            ## generate conditional probabilities for ghost moves
            ## Expectation function: Pacman's beliefs of which moves the ghost will take
            ##
            ## Equation: Expectation = sum([score * P(Move=move) for move,score in scores_with_different_moves])
            ##      move: categorical variable coded as ... 
            ##      score: the score of the game if the ghost makes that move
            ##      Expectation: the weighted average score given how the ghosts move
            ##
            ## This function WOULD USUALLY generate the P(Move=move) probabilities from prior data and generated data
            ## The full expectation equation is coded in the ghostValue() function; ghostValue() returns the average score.
            ##
            ## HOWEVER, we're just assuming we'll be encountering RandomGhosts with a uniform distribution of probabilities...
            
            p  = {m:1./len(moves) for m in moves}
            return p

        selected = value(gameState)
        return selected[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    curScore = currentGameState.getScore() #this is to get already eaten food and time penalty factor
    problem = FoodSearchProblem(currentGameState) ## convert the current game state to a simpler version of the state space
    foodUtility = foodHeuristic(problem) #apply the aStarSearch to get utility of eating all food from current position
    ghostsDist = ghostHeuristic(problem) #apply the aStarSearch to get the distances to all ghosts as sum
    
    #TODO: write code to finish these
    scaredGhostUtility = scaredGhostHeuristic(problem)
    capsuleUtility = capsuleHeuristic(problem)

    #TODO: recalibrate these once scaredGhostUtility and capsuleUtility functions are added
    #      (Are we overfitting these to the current game?)
    foodWeight = 1
    ghostWeight = 4.2
    scaredGhostWeight = 1
    capsuleWeight = 1

    output = curScore + \
            foodUtility*foodWeight + \
            ghostsDist*ghostWeight + \
            scaredGhostUtility*scaredGhostWeight + \
            capsuleUtility*capsuleWeight  
    #print(output)
    return output
    

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

############################
## modified code from searchAgents.py
############################

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood()) ##this is the current state information
        self.ghosts = startingGameState.getGhostPositions() ##NEW: for ghost heuristic
        self.capsules = startingGameState.getCapsules()
        self.ghost_states = startingGameState.getGhostStates()
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

def foodHeuristic(problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, problem.walls gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use. For example,
    if you only want to count the walls once and store that value, try:
      problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount']
    """
    state = problem.start ##the FSP will have this info
    foodGrid = state[1].data

    # goals = [x for x in foodGrid if x is True]

    goals = set()
    for rowIndex, row in enumerate(foodGrid):
        for columnIndex, column in enumerate(row):
            if column:
                goals.add((rowIndex, columnIndex))

    # position = state

    # distances = list()
    distances = []
    current_sum = 0
    min_dist = float("inf")
    min_point = None
    if len(goals) == 0: return 0
    for x, y in goals:
        d = abs(state[0][0] - x) + abs(state[0][1] - y)
        if d < min_dist:
            min_dist = d
            min_point = (x, y)
    if min_point is None:
        raise Exception("No closest point found.")

    #print("closest point to goal: " + str(min_point))
    current_sum += min_dist
    distances.append(1./current_sum)
    remaining = goals
    remaining.remove(min_point)
    atpoint = min_point
    while (remaining):
        min_dist = float("inf")
        min_point = None
        for x, y in remaining:
            d = abs(atpoint[0] - x) + abs(atpoint[1] - y)
            if d < min_dist:
                min_dist = d
                min_point = (x, y)
        current_sum += min_dist
        distances.append(1./current_sum)
        remaining.remove(min_point)
        atpoint = min_point
    
    #utility function: 10 points for each pellet of food to eat, -5 for each space Pacman has to move
    return 10*len(distances) - 5*current_sum #add ten points for each pellet eaten, subtract five points time penalty for distance to food

## NEW: based on foodHeuristic
def ghostHeuristic(problem):
    """
    finds closest distances for all ghosts
    """
    state = problem.start ##the FSP will have this info

    currentGhostStates = problem.ghost_states
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    goals = [ghost.getPosition() for ghost in currentGhostStates if ghost.scaredTimer <= 1]
    goals = set(goals)


    # position = state

    # distances = list()
    #distances = []
    current_sum = 0
    min_dist = float("inf")
    min_point = None
    if len(goals) == 0: return 0

    for x, y in goals:
        d = abs(state[0][0] - x) + abs(state[0][1] - y)
        if d < min_dist:
            min_dist = d
            min_point = (x, y)
    if min_point is None:
        raise Exception("No closest point found.")

    #print("closest point to goal: " + str(min_point))
    current_sum += min_dist
    #distances.append(1./current_sum)
    remaining = goals
    remaining.remove(min_point)
    atpoint = min_point
    while (remaining):
        min_dist = float("inf")
        min_point = None
        for x, y in remaining:
            d = abs(atpoint[0] - x) + abs(atpoint[1] - y)
            if d < min_dist:
                min_dist = d
                min_point = (x, y)
        current_sum += min_dist
        #distances.append(1./current_sum)
        remaining.remove(min_point)
        atpoint = min_point
    
    #return sum of distances to all ghosts (higher is better for fearless ghosts)
    return current_sum


def scaredGhostHeuristic(problem):
    """
    finds closest distances for all ghosts
    """
    state = problem.start  ##the FSP will have this info
    #ghostPositions = problem.ghosts  ##change from foodGrid to ghost positions as goals

    #goals = set(ghostPositions)

    # position = state

    # distances = list()
    # distances = []
    current_sum = 0
    min_dist = float("inf")
    min_point = None

    currentGhostStates = problem.ghost_states
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    goals = [ghost.getPosition() for ghost in currentGhostStates if ghost.scaredTimer > 1]
    goals = set(goals)

    if len(goals) == 0: return 0


    for x, y in goals:
        d = abs(state[0][0] - x) + abs(state[0][1] - y)
        if d < min_dist:
            min_dist = d
            min_point = (x, y)
    if min_point is None:
        raise Exception("No closest point found.")


    # print("closest point to goal: " + str(min_point))
    current_sum += min_dist
    # distances.append(1./current_sum)
    remaining = goals
    remaining.remove(min_point)
    atpoint = min_point
    while (remaining):
        min_dist = float("inf")
        min_point = None
        for x, y in remaining:
            d = abs(atpoint[0] - x) + abs(atpoint[1] - y)
            if d < min_dist:
                min_dist = d
                min_point = (x, y)
        current_sum += min_dist
        # distances.append(1./current_sum)
        remaining.remove(min_point)
        atpoint = min_point

    # return sum of distances to all ghosts (higher is better for fearless ghosts)
    return 100*len(goals) - 5*current_sum #Return 100 * the number of scared ghosts - 5 * the steps wasted not gaining points


def capsuleHeuristic(problem):

    state = problem.start  ##the FSP will have this info
    goals = set(problem.capsules)
    # position = state

    # distances = list()
    distances = []
    current_sum = 0
    min_dist = float("inf")
    min_point = None
    if len(goals) == 0: return 0
    for x, y in goals:
        d = abs(state[0][0] - x) + abs(state[0][1] - y)
        if d < min_dist:
            min_dist = d
            min_point = (x, y)
    if min_point is None:
        raise Exception("No closest point found.")

    # print("closest point to goal: " + str(min_point))
    current_sum += min_dist
    distances.append(1. / current_sum)
    remaining = goals
    remaining.remove(min_point)
    atpoint = min_point
    while (remaining):
        min_dist = float("inf")
        min_point = None
        for x, y in remaining:
            d = abs(atpoint[0] - x) + abs(atpoint[1] - y)
            if d < min_dist:
                min_dist = d
                min_point = (x, y)
        current_sum += min_dist
        distances.append(1. / current_sum)
        remaining.remove(min_point)
        atpoint = min_point

    # utility function: 10 points for each pellet of food to eat, -5 for each space Pacman has to move
    return 10 * len(distances) - 20 * current_sum  # add ten points for each pellet eaten, subtract five points time penalty for distance to food
