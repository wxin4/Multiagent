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
import collections

from util import manhattanDistance
from game import Directions
import random, util
import random
import bisect

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # initialize a manhattan distance to hold the distance between the new position of pacman and the current food position
        man_dist = []

        # if pacman stops, it is a bad move and may cause point deduction
        if action == 'Stop':
            return float('-inf')

        # loop through the ghost state and find if the new position of the pacman is in the ghost position, meanwhile, the ghost is not in its scared time
        # then the score will be deducted since it meets a not-scared ghost
        for ghost_state in newGhostStates:
            if newPos == ghost_state.getPosition() and ghost_state.scaredTimer <= 0:
                return float('-inf')

        # loop through the food positions and calculate the distance between new position of pacman and the food
        foodList = oldFood.asList()
        for foodPos in foodList:
            dist = manhattanDistance(newPos, foodPos)
            # As features, try the reciprocal of important values (such as distance to food) rather than just the values themselves
            man_dist.append(dist * (-1))  # here we multiply the distance by (-1) is because we have a note above

        # get the maximum manhattan distance of a negative number which is the closest food for the pacman
        return max(man_dist)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


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
                # if the agent is in the last position of the agents list, meaning we should go to the next level of the tree and reset it to the pacman agent
                if agentIndex == gameState.getNumAgents() - 1:
                    mini = self.max_value(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)

                # else, continue traversing the ghost optimal tree
                else:
                    mini = self.min_value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)

                # if the previous min value is not the minimum value anymore, replace it
                if mini < v:
                    v = mini

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
        # simply call the value function to start the minimax traversing.

        # weights = [0.8, 0.05, 0.05, 0.05, 0.05]
        # states = ['North', 'South', 'East', 'West', 'Stop']
        # expected_value = self.value(gameState, 0, 0)
        # states.remove(expected_value)
        # population = [expected_value, states[0], states[1], states[2], states[3]]
        # counts = collections.defaultdict(int)
        # for i in range(10000):
        #     counts[self.choice(population, weights)] += 1
        # print(counts)
        # print(self.value(gameState, 0, 0))
        if 'Left' in gameState.getLegalActions(0) or 'Right' in gameState.getLegalActions(0) or 'Center' in gameState.getLegalActions(0):
            weights = [0.8, 0.1, 0.1]
            states = ['Left', 'Right', 'Center']
            expected_value = self.value(gameState, 0, 0)
            states.remove(expected_value)
            population = [expected_value, states[0], states[1]]
            counts = collections.defaultdict(int)
            for i in range(10000):
                counts[self.choice(population, weights)] += 1
            print(counts)

        elif 'North' in gameState.getLegalActions(0) or 'South' in gameState.getLegalActions(0) or 'East' in gameState.getLegalActions(0) or 'West' in gameState.getLegalActions(0) or 'Stop' in gameState.getLegalActions(0):
            weights = [0.8, 0.05, 0.05, 0.05, 0.05]
            states = ['North', 'South', 'East', 'West', 'Stop']
            expected_value = self.value(gameState, 0, 0)
            states.remove(expected_value)
            population = [expected_value, states[0], states[1], states[2], states[3]]
            counts = collections.defaultdict(int)
            for i in range(10000):
                counts[self.choice(population, weights)] += 1
            print(counts)

        return self.value(gameState, 0, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    # define the value function showed in the slides
    def value(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # if the agent index is 0, meaning it is pacman optimal
        if (agentIndex == 0):
            return self.pruneMax(gameState, depth, agentIndex, alpha, beta)

        # ghost optimal
        else:
            return self.pruneMin(gameState, depth, agentIndex, alpha, beta)

    # max value function passing the depth and agent index
    def pruneMax(self, gameState, depth, agentIndex, alpha, beta):
        # initialize the current value as negative infinity
        v = float("-inf")

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
                maxi = self.pruneMin(child, depth, agentIndex + 1, alpha, beta)

                # HERE IS THE ADDED PART 1
                if maxi > beta:
                    return maxi

                # if the max value is greater than the stored max value, replace it
                if maxi > v:
                    v = maxi
                    result = action  # for the getAction function use.

                # HERE IS THE ADDED PART 2
                alpha = max(alpha, v)

        # if the state reaches to itself, meaning the state depth is 0 and it should be the pacman's state (max_value)
        if depth == 0:
            return result
        else:
            return v  # if not, continue by passing the maximum value

    # max value function passing the depth and agent index
    def pruneMin(self, gameState, depth, agentIndex, alpha, beta):
        v = float("inf")
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        else:
            action_list = gameState.getLegalActions(agentIndex)
            for action in action_list:
                # if the agent is in the last position of the agents list, meaning we should go to the next level of the tree and reset it to the pacman agent
                if agentIndex == gameState.getNumAgents() - 1:
                    mini = self.pruneMax(gameState.generateSuccessor(agentIndex, action), depth + 1, 0, alpha, beta)

                # else, continue traversing the ghost optimal tree
                else:
                    mini = self.pruneMin(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta)

                # HERE IS THE ADDED PART 1
                if mini < alpha:
                    return mini

                # if the previous min value is not the minimum value anymore, replace it
                if mini < v:
                    v = mini

                # HERE IS THE ADDED PART 2
                beta = min(beta, v)

        return v

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # simply call the value function to start the minimax traversing.
        alpha = float('-inf')
        beta = float('inf')
        return self.value(gameState, 0, 0, alpha, beta)


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
                # if the agent is in the last position of the agents list, meaning we should go to the next level of the tree and reset it to the pacman agent
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
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:  The whole score is based on the pacman to the nearest food distance and to the nearest ghost distance. If the pacman is closer to the food,
                    increase the score based on the real distance, meaning if the pacman is 3 steps from the food, the score is higher than the 4 steps away.
                    Additionally, the score is higher if the pacman is moving away from the ghost. If the pacman has a closer distance to the ghost, the score
                    will be deducted. Obviously, if the ghost is not in its paralyzed state, the score will be lower than the opposite state. I consider the
                    scared time as well. The code below is based on the real test using the autograder of how to set up the "20", "150", "100" and so on. Really
                    easy to go thru!

      Scores shown in the autograder:
                    Pacman emerges victorious! Score: 905
                    Pacman emerges victorious! Score: 945
                    Pacman emerges victorious! Score: 1052
                    Pacman emerges victorious! Score: 1097
                    Pacman emerges victorious! Score: 1332
                    Pacman emerges victorious! Score: 1020
                    Pacman emerges victorious! Score: 1071
                    Pacman emerges victorious! Score: 857
                    Pacman emerges victorious! Score: 1101
                    Pacman emerges victorious! Score: 1342
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

    # return the whole score (NOTE THAT THE NUMBERS ARE ALL TESTED BASED ON THE AUTOGRADER IN ORDER TO GET FULL CREDIT)
    return currentGameState.getScore() + foodDistScore + ghostDistScore


# Abbreviation
better = betterEvaluationFunction
