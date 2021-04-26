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


        states = mdp.getStates()
        Val = self.values

        #Val iteration function:
        #value of state = max action{sum over outcome states{prob of ending in state * reward from taking action + discountVal * val state end up in}}

        for i in range(0, self.iterations): #Update value until we get to the last iteration
            Valplus1 = copy.copy(Val) #Make a copy of the values so we can update it later, offline
            for s in states: #for each state
                actions = mdp.getPossibleActions(s) #Get list of actions you can from current state
                expected_values = list() #make a list to hold the sums for each action
                for a in actions: #for each action
                    transitionStatesAndProbs = mdp.getTransitionStatesAndProbs(s, a)  # Gives tuples of (newState, probability of ending in state)
                    current_sum = 0 #initialize variable to hold sumOutcome{probEndingInState * rewardFromAction + discountVal * outcomeStateVal}
                    for outcome, prob in transitionStatesAndProbs: #for each outcome and probability
                        current_sum += prob * (mdp.getReward(s, a, outcome) + (self.discount * Val[outcome])) #sum over outcomes
                    expected_values.append(current_sum) #store all the current sums influenced by each action
                if(len(expected_values) == 0 ): #if nothing was added to expected_values list, then there is no value
                    Valplus1[s] = 0
                else:
                    Valplus1[s] = max(expected_values) #this finds the highest sum resulting from a certain action
            Val = Valplus1 #Update value

        self.values = Val #update values




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

        #We have one state, one iteration this time
        #same equation as above, but without max action part

        #Q(state, action) = {sum over outcome states{prob of ending in state * reward from taking action + discountVal * val state end up in}

        mdp = self.mdp
        Val = self.values

        transitionStatesAndProbs = mdp.getTransitionStatesAndProbs(state, action)  # Gives tuples of (newState, probability)
        current_sum = 0 #variable to old sum over outcomes of the equation
        for outcome, prob in transitionStatesAndProbs:
            current_sum += prob * (mdp.getReward(state, action, outcome) + (self.discount * Val[outcome]))  # sum over outcomes like before

        return current_sum #just return the sum



    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        #We have one state, one iteration
        #policy(state) = arg max action{Q(state,action)}
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)  # Get list of actions you can from current state
        Val = self.values
        expected_values = list() #will hold Q(state, action) and action that led to that


        for a in actions: #traverse through actions
            transitionStatesAndProbs = mdp.getTransitionStatesAndProbs(state,a)  # Gives tuples of (newState, probability)
            current_sum = 0
            for outcome, prob in transitionStatesAndProbs: #We will calculate Q(state, action) and store in current_sum
                current_sum += prob * (mdp.getReward(state, a, outcome) + (self.discount * Val[outcome]))  # sum over outcomes

            expected_values.append((current_sum,a))  # add current sum for certain action to our list
        if (len(expected_values) == 0): #if nothing added to list, return None
            return None
        else:
            maxi = max(expected_values)  # Take max of all actions
            return maxi[1] #return action that led to highest Q(state,action)







    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
