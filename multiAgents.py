# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import bisect
import time as timelibrary

from game import Agent


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


    def __init__(self, withprobflag, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.betterEvaluationFunction = util.lookup(evalFn, globals())
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.withprobflag = withprobflag
        

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    # define the value function showed in the slides
    def value(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # if the agent index is 0, meaning it is pacman optimal
        if agentIndex == 0:
            return self.max_value(gameState, depth, agentIndex)

        # ghost optimal
        else:
            return self.min_value(gameState, depth, agentIndex)

    # max value function passing the depth and agent index
    def max_value(self, gameState, depth, agentIndex):
        # initialize the current value as negative infinity
        v = float('-inf')

        # condition check: if it is already in the win or lose or the depth is its current depth,
        # go the the evaluation function and calculate the distance
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        else:
            # loop through the actions of the agent (pacman)
            action_list = gameState.getLegalActions(agentIndex)
            for action in action_list:
                child = gameState.generateSuccessor(agentIndex, action)  # the child (successor)

                # pass the min_value function by incrementing agent to ghost
                maxi = self.min_value(child, depth, agentIndex + 1)

                # if the max value is greater than the stored max value, replace it
                if maxi > v:
                    v = maxi
                    result = action  # for the getAction function use.

        # if the state reaches to itself, meaning the state depth is 0 and it should be the pacman's state (max_value)
        if depth == 0:
            return result
        else:
            return v  # if not, continue by passing the maximum value

    # max value function passing the depth and agent index
    def min_value(self, gameState, depth, agentIndex):
        v = float('inf')
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        else:
            action_list = gameState.getLegalActions(agentIndex)
            for action in action_list:
                # if the agent is in the last position of the agents list, meaning we should go to the next level of
                # the tree and reset it to the pacman agent
                if agentIndex == gameState.getNumAgents() - 1:
                    mini = self.max_value(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)

                # else, continue traversing the ghost optimal tree
                else:
                    mini = self.min_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

                # if the previous min value is not the minimum value anymore, replace it
                if mini < v:
                    v = mini
                    result = action

        return v

    def cdf(self, weights):
        total = sum(weights)
        result = []
        cumsum = 0
        for w in weights:
            cumsum += w
            result.append(cumsum / total)
        return result

    def choice(self, population, weights):
        assert len(population) == len(weights)
        cdf_vals = self.cdf(weights)
        x = random.random()
        idx = bisect.bisect(cdf_vals, x)
        return population[idx]

    def getAction(self, gameState):
        '''
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
        '''
        "*** YOUR CODE HERE ***"
        # simply call the value function to start the minimax traversing.

        result = ''
        legal_states_pacman = gameState.getLegalActions(0)

        if 'Left' in gameState.getLegalActions(0) or 'Right' in gameState.getLegalActions(0) or 'Center' in gameState.getLegalActions(0):
            weights = [0.8]
            states = [self.value(gameState, 0, 0)]
            legal_states_pacman.remove(self.value(gameState, 0, 0))
            for ls in legal_states_pacman:
                weights.append(0.2 / len(legal_states_pacman))
                states.append(ls)

            result = self.choice(states, weights)

        elif 'North' in gameState.getLegalActions(1) or 'South' in gameState.getLegalActions(1) or 'East' in gameState.getLegalActions(1) or 'West' in gameState.getLegalActions(1) or 'Stop' in gameState.getLegalActions(1):
            weights = [0.8]
            states = [self.value(gameState, 0, 0)]
            legal_states_pacman.remove(self.value(gameState, 0, 0))
            for ls in legal_states_pacman:
                weights.append(0.2 / len(legal_states_pacman))
                states.append(ls)

            result = self.choice(states, weights)

        return result


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # define the value function showed in the slides
    def value(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # if the agent index is 0, meaning it is pacman optimal
        if (agentIndex == 0):
            return self.max_value(gameState, depth, agentIndex)

        # ghost optimal
        else:
            return self.expect_value(gameState, depth, agentIndex)

    # max value function passing the depth and agent index
    def max_value(self, gameState, depth, agentIndex):
        # initialize the current value as negative infinity
        v = float('-inf')

        # condition check: if it is already in the win or lose or the depth is its current depth,
        # go the the evaluation function and calculate the distance
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        else:
            # loop through the actions of the agent (pacman)
            action_list = gameState.getLegalActions(agentIndex)
            for action in action_list:
                child = gameState.generateSuccessor(agentIndex, action)  # the child (successor)

                # pass the min_value function by incrementing agent to ghost
                maxi = self.expect_value(child, depth, agentIndex + 1)

                # if the max value is greater than the stored max value, replace it
                if maxi > v:
                    v = maxi
                    result = action  # for the getAction function use.

        # if the state reaches to itself, meaning the state depth is 0 and it should be the pacman's state (max_value)
        if depth == 0:
            return result
        else:
            return v  # if not, continue by passing the maximum value

    # expect value function passing the depth and agent index
    def expect_value(self, gameState, depth, agentIndex):
        # initialize v equals to 0 as the slide shown
        v = 0
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        else:
            action_list = gameState.getLegalActions(agentIndex)
            for action in action_list:
                # if the agent is in the last position of the agents list, meaning we should go to the next level of
                # the tree and reset it to the pacman agent
                if agentIndex == gameState.getNumAgents() - 1:
                    v += self.max_value(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)

                # else, continue traversing the ghost optimal tree
                else:
                    v += self.expect_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

        return v / len(action_list)  # This is the result with the probability

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # simply call the value function to start the minimax traversing.
        return self.value(gameState, 0, 0)


def betterEvaluationFunction(currentGameState):
    """
    """
     
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState
    currPos = currentGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    # initialize the pacman to the food and ghost distance
    foodDistScore = 0
    ghostDistScore = 0
    newFoodList = newFood.asList()

    # get the minimum distance from the current position of pacman to the new food position
    min_dist_from_curr_pos_to_food = min(0, (manhattanDistance(currPos, new_food) for new_food in newFoodList))

    # the food distance score is based on the distance from pacman to food
    if min_dist_from_curr_pos_to_food != 0:
        foodDistScore = 20 / min_dist_from_curr_pos_to_food  # if the pacman is closer to the food, the score is high

    # get the ghost position and pacman to ghost distance
    ghostPos = currentGameState.getGhostPositions()[0]
    pac_to_ghost_dist = manhattanDistance(ghostPos, currPos)

    # if the distance to the ghost is greater than 4 and the ghost is in the paralyzing position, the score increases based on the distance
    if pac_to_ghost_dist >= 4 and newScaredTimes[0] > 0:
        ghostDistScore += 100 / pac_to_ghost_dist

    # if the distance is 0 and the ghost is not paralyzed, meaning the pacman is caught, decrease the score big time!
    elif pac_to_ghost_dist == 0 and newScaredTimes[0] == 0:
        ghostDistScore -= 150

    # else, meaning the pacman is in danger, we need to decrease the score by the distance
    else:
        ghostDistScore -= 20 / pac_to_ghost_dist

    currenttime = timelibrary.time()

    timelapsed = (currenttime - currentGameState.data.gametimestart)*10
    # print timelapsed
    # print current
    # print currenttime - currentGameState.data.gametimestart, "VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV"
    # return the whole score (NOTE THAT THE NUMBERS ARE ALL TESTED BASED ON THE AUTOGRADER IN ORDER TO GET FULL CREDIT)
    return currentGameState.getScore() + foodDistScore + ghostDistScore





# Abbreviation
better = betterEvaluationFunction
