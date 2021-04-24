"LAURA NEFF"
"laura.neff@emory.edu/lneff2/2304477"
"THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING "
"ANY# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. LAURA NEFF"

# valueIterationAgents.py
# -----------------------
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


import mdp, util, copy
# import numpy as np
# from itertools import chain

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = mdp.getStates()
        Val = self.values
        # oldVal = 0

        for i in range(0, self.iterations):
            Valplus1 = copy.copy(Val)
            for x in states:
                action = mdp.getPossibleActions(x) #Get list of actions you can from current state
                expected_values = list()
                for a in action:
                    transitionStatesAndProbs = mdp.getTransitionStatesAndProbs(x, a)  # Gives tuples of (newState, probability of ending in state)
                    # outcomes = [i[0] for i in transitionStatesAndProbs]  # Only gets keys (newState) from tuple (newState, probability of ending in state)
                    # probabilities = [i[1] for i in transitionStatesAndProbs]  # probability of ending in state
                    current_sum = 0
                    # for y in outcomes:
                    #     for z in probabilities:
                    for y, z in transitionStatesAndProbs:
                        current_sum += z * (mdp.getReward(x, action, y) + (self.discount * Val[y])) #sum over outcomes
                    expected_values.append(current_sum) #add current sum for iteration for certain action
                if(len(expected_values) == 0 ):
                    Valplus1[x] = 0
                else:
                    Valplus1[x] = max(expected_values) #Take max of all actions
            Val = Valplus1 #Update values

        self.values = Val

                #sum over outcomes
                #take max of actions... but how?


                # oldVal = Val[x]

        # for x in states:
        #     action = mdp.getPossibleActions(x)
        #     transitionState = mdp.getTransitionStatesAndProbs(x, action)
        #     print(transitionState)


        # print(self.values)
        # for x in states:
            #Val.append(0)
            # print(x)



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #We have one state, one iteration
        mdp = self.mdp
        Val = self.values

        transitionStatesAndProbs = mdp.getTransitionStatesAndProbs(state, action)  # Gives tuples of (newState, probability)
        current_sum = 0
        for y, z in transitionStatesAndProbs:
            current_sum += z * (mdp.getReward(state, action, y) + (self.discount * Val[y]))  # sum over outcomes

        return current_sum


        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #We have one state, one iteration
        mdp = self.mdp
        action = mdp.getPossibleActions(state)  # Get list of actions you can from current state
        Val = self.values
        expected_values = list()

        for a in action:
            transitionStatesAndProbs = mdp.getTransitionStatesAndProbs(state,a)  # Gives tuples of (newState, probability)
            current_sum = 0
            for y, z in transitionStatesAndProbs:
                current_sum += z * (mdp.getReward(state, a, y) + (self.discount * Val[y]))  # sum over outcomes

            expected_values.append((current_sum,a))  # add current sum for certain action
        if (len(expected_values) == 0):
            return None
        else:
            maxi = max(expected_values)  # Take max of all actions
            return maxi[1]





        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
