# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util, copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

class Node:
    ## defines a node on a vertex

    def __init__(self, vertex, problem, path_to=[], path_cost=0, heuristic=0, f= 0):
        self.vertex = vertex
        self.path_to = path_to
        self.path_cost = path_cost
        self.problem = problem
        self.heuristic = heuristic
        self.f = f

    def getSuccessors(self):
        return self.problem.getSuccessors(self.vertex)

    def isGoalState(self):
        return self.problem.isGoalState(self.vertex)

    def __hash__(self):
        return hash(self.vertex)

    def __eq__(self, other):
        if type(other) != type(self): return False
        return self.vertex == other.vertex

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #use only the methods here
    #Only Pacman's position matters
    #problem = object of SearchProblem

    #Add corner case here -- is start state = goal state?

    traversed = set()
    stack = util.Stack()
    initial_vertex = Node(problem.getStartState(), problem) #initial vertex

    #initial_vertex.path_to = []

    stack.push(initial_vertex) #stores initial vertex

    #neighbors = [neighbor for neighbor, action, cost in problem.getSuccessors()]

    while (not stack.isEmpty()):
        current = stack.pop()
        if (current.isGoalState()): return current.path_to  # be sure to check this before expanding the state
        if(current not in traversed):

            neighbors_actions = [(Node(neighbor, problem), action) for neighbor, action, cost in current.getSuccessors()]
            traversed.add(current)

            for neighbor, action in neighbors_actions: #check the whole graph and mark the traversed ones
                    neighbor.path_to = copy.copy(current.path_to)
                    neighbor.path_to.append(action)
                    stack.push(neighbor) #push the untraversed neighbors to the stack

    return None

    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    traversed = set()
    queue = util.Queue()
    initial_vertex = Node(problem.getStartState(), problem)  # initial vertex
    queue.push(initial_vertex)  # stores initial vertex

    while (not queue.isEmpty()):
        current = queue.pop()
        if (current.isGoalState()): return current.path_to  # be sure to check this before expanding the state
        if(current not in traversed):

            neighbors_actions = [(Node(neighbor, problem), action) for neighbor, action, cost in current.getSuccessors()]
            traversed.add(current)

            for neighbor, action in neighbors_actions: #check the whole graph and mark the traversed ones
                    neighbor.path_to = copy.copy(current.path_to)
                    neighbor.path_to.append(action)
                    queue.push(neighbor) #push the untraversed neighbors to the stack

    return None



    #util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    traversed = set()
    priority_queue = util.PriorityQueue()
    initial_vertex = Node(problem.getStartState(), problem)  # initial vertex
    priority_queue.push(initial_vertex, initial_vertex.path_cost)  # stores initial vertex. Priority = path cost
    #item, priority = PriorityQueue constructor

    while (not priority_queue.isEmpty()):
        current = priority_queue.pop()
        if current in traversed:
            continue  #check if we already saw the item

        if (current.isGoalState()):
            return current.path_to

        traversed.add(current)

        neighbors_actions_cost = [(Node(neighbor, problem), action, cost) for neighbor, action, cost in current.getSuccessors()]

        for neighbor, action, cost in neighbors_actions_cost: #check the whole graph and mark the traversed ones
            neighbor.path_to = copy.copy(current.path_to)
            neighbor.path_to.append(action)
            neighbor.path_cost = current.path_cost + cost

            if (neighbor not in traversed):
                priority_queue.push(neighbor, neighbor.path_cost)

            if(current not in traversed or priority_queue.heap):
                priority_queue.push(current, current.path_cost)


    return None


    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"

    untraversed = util.PriorityQueue()
    traversed = set()


    #f = h + g --> total cost from current node to goal node
    #g = path cost = original node to current node
    #h = heuristic --> estimated guess of cost of current node to goal node

    # item, priority = PriorityQueue constructor

    initial_vertex = Node(problem.getStartState(), problem)  # initial vertex

    untraversed.push(initial_vertex, initial_vertex.f)  # stores initial vertex. Priority = path cost
    #We are prioritizing by f = g + h!!

    while (not untraversed.isEmpty()):
        current = untraversed.pop()
        if current in traversed:
            continue  # check if we already saw the item

        if (current.isGoalState()):
            return current.path_to

        traversed.add(current)

        neighbors_actions_cost = [(Node(neighbor, problem), action, cost) for neighbor, action, cost in
                                  current.getSuccessors()]

        for neighbor, action, cost in neighbors_actions_cost:  # check the whole graph and mark the traversed ones
            neighbor.path_to = copy.copy(current.path_to)
            neighbor.path_to.append(action)
            neighbor.path_cost = current.path_cost + cost #This is g
            neighbor.heuristic = heuristic(neighbor.vertex, problem) #this is h
            neighbor.f = neighbor.heuristic + neighbor.path_cost

            if (neighbor not in traversed):
                untraversed.push(neighbor, neighbor.f)

            if (current not in traversed or untraversed.heap):
                untraversed.push(current, current.f)

    return None




    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
