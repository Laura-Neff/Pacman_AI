
"LAURA NEFF"
"laura.neff@emory.edu/lneff2/2304477"
"THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING "
"ANY# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. LAURA NEFF"

# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)


        self.Q = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """

        # return the util.Counter() object self.Q for certain state and action
        return self.Q[(state, action)]



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        #Find best Q-value that is the result of an action

        actions = self.getLegalActions(state) #find all legal actions for state
        if(len(actions) == 0): #If the legalActions list is empty, return 0.0
            return 0.0
        else:
            Q = list() #initialize list to hold Q-values
            for a in actions: #traverse through actions
                Q.append((self.getQValue(state, a), a)) #store Q-values for each action and record to the action linked to it

            maxi = max(Q)  # Take max of all Q-vals
            return maxi[0] #return Q-val







    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """

        # Find best action to obtain the best Q-value
        actions = self.getLegalActions(state) #find all legal actions for state
        if (len(actions) == 0): #if no legal actions for state, return 0.0
            return 0.0
        else:
            Q = list() #initialize list to hold Q-vals
            for a in actions: #traverse actions
                Q.append((self.getQValue(state, a), a)) #store Q-val for each action and record the action linked to it

            maxi = max(Q)  # Take max of all Q-vals
            return maxi[1] #return action that led to max Q-val




    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if(util.flipCoin(self.epsilon)): #If probability is determined to be epsilon
            action = random.choice(legalActions) #Choose a random action
        else: #If probability is anything other than epsilon
            action = self.computeActionFromQValues(state) #Return best policy action


        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """

        #sample = reward + discount * Q(state,action)
        #Q(state,action) = ((1 - alpha) * (Q(state,action)) + (alpha * sample) where alpha = learning coefficient
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.Q[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample



    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())() #feat = feature representation, this code finds function
                                                                 # according to features (feature extraction function)
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        #Q(state,action) = w1 * f1(state,action) + w2 * f2(state,action) + ...  wn * fn(state,action)

        features = self.featExtractor.getFeatures(state,action) #Returns a dict from features to counts
        Q = 0 #initialize Q(state,action) to 0
        for f in features.keys(): #Find all keys in feature function dictionary because key is shared with weight
                                  #Depending on function, you will get a particular weight
           Q += self.weights[f] * features[f] #Do formula described at top of method

        return Q


    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        #difference = [reward + discountVal * max outcomeActions{Q(outcomeState,outcomeAction)}] - Q(state,action)
        #weight = weight + [alpha * difference * function(state,action)]

        actions = self.getLegalActions(nextState) #find all legal actions of the nextState/outcome state
        if (len(actions) == 0):  # If the legalActions list is empty, return 0
            maximum = 0.0
        else:
            Q = list() #initialize list to hold Q-vals
            for a in actions: #traverse through all legal actions of next state
                Q.append((self.getQValue(nextState, a), a)) #store all Q-vals according to each outcome action and record linked outcome action

            maxi = max(Q)  # Take max of all Q-vals
            maximum = maxi[0] #record max Q-val


        #Now bring formulas at top of method together to do approximate Q-val update
        difference = (reward + self.discount * maximum) - self.getQValue(state, action)

        features = self.featExtractor.getFeatures(state, action)
        for f in features.keys():
            self.weights[f] = self.weights[f] + (self.alpha * difference) * features[f]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
