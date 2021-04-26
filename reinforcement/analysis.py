"LAURA NEFF"
"laura.neff@emory.edu/lneff2/2304477"
"THIS CODE WAS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING "
"ANY# SOURCES OUTSIDE OF THOSE APPROVED BY THE INSTRUCTOR. LAURA NEFF"

# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    #When the agent thinks more about long-term gains and the agent ends up where it is expected most of the time
    #the optimal policy causes the agent to cross the bridge
    answerDiscount = 0.9
    answerNoise = .001
    return answerDiscount, answerNoise

def question3a():
    #When agent thinks about long-term gains, ends up where expected, and is highly motivated to reach goal
    #it prefers the close exit and risks the cliff
    answerDiscount = .9 #Agent cares about long-term or short-term gains?
    answerNoise = .01 #How often do we get something different than we expected? Randomness vs. expected
    answerLivingReward = -5 #The pressure of reaching our goal
    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    #When the agent only cares about short-term gains, ends up where expected, and isn't pressured all that much to
    #reach goal, it prefers the close exit but avoids the cliff
    answerDiscount = .01 #Thinks about short-term gains
    answerNoise = .01 #Agent gets where expected
    answerLivingReward = -1 #No pressure to reach goal -- can be careful
    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    #When agent thinks long term, gets where expected, and isn't pressured to reach goal, it prefers the distant exit
    #and risks the cliff
    answerDiscount = .9 #thinks long-term
    answerNoise = .01 #agent gets where expected
    answerLivingReward = -1 #Low pressure to reach goal
    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    #when agent thinks about long-term gains, doesn't end up where expected a lot, and isn't pressured to reach goal,
    #it prefers the distant exit and avoids the cliff
    answerDiscount = .9 #thinks long-term
    answerNoise = .5 #agent doesn't end up where expected a lot
    answerLivingReward = -1 #not pressured to reach goal
    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    #when agent thinks about long-term gains, ends up where expected most the time, and there's a huge reward for reaching
    #the goal, it avoids both exits and the cliff so an episode should never terminate
    answerDiscount = .9 #thinks about long-term gains
    answerNoise = .01 #agent ends where expected most the time
    answerLivingReward = 5 #huge reward for reaching goal
    return answerDiscount, answerNoise, answerLivingReward

def question6():
    #There is no epsilon and learning rate such that it is greater than 99% that the optimal policy
    #will be learned after 50 iterations

    #I will keep these variables to represent that there is no such epsilon and learning rate
    #to the reader of the code
    answerEpsilon = 0
    answerLearningRate = 0
    return "NOT POSSIBLE"

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
