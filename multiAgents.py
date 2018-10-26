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
import random, util, sys

from game import Agent
debug = False

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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ghostDistance = list()
        foodDistance = list()
        totalFood = 0
        heuristic = 0

        for food in newFood:
            for num in food:
                totalFood += int(num)

        heuristic -= 100 * totalFood

        for ghost in newGhostStates:
            ghostDistance += [manhattanDistance(ghost.getPosition(), newPos)]
        for food in newFood.asList():
            foodDistance = foodDistance + [manhattanDistance(newPos, food)]
            

        if newGhostStates and totalFood:
            heuristic -= min(foodDistance)

            if min(ghostDistance) == 0:
                heuristic -= 9999
            else:
                heuristic -= 10 / min(ghostDistance)
        return heuristic        


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
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & maxPlayerPacmanAgent.

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

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        ghostCount = gameState.getNumAgents() - 1

        def maxPlayer(state, depth):
          if depth == 0:
            return [self.evaluationFunction(state), None]
          if state.getLegalActions(0) == []:
            return [self.evaluationFunction(state), None]
          v = [-float('inf'), None]
          for action in state.getLegalActions(0):
            result = maxHelper(v, minPlayer(state.generateSuccessor(0, action), 1, depth), action)
          return result
            
        def maxHelper(v, minResult, action):
          if minResult[0] > v[0]:
            v[0] = minResult[0]
            v[1] = action
          return v

        def minPlayer(state, ghostCount, depth):
          if state.getLegalActions(ghostCount) == []:
            return [self.evaluationFunction(state), None]
          v = [float('inf'), None]
          for action in state.getLegalActions(ghostCount):
            if ghostCount != state.getNumAgents() - 1:
              result = minHelper(v, minPlayer(state.generateSuccessor(ghostCount, action), ghostCount + 1, depth), action)
            else:
              result = minHelper(v, maxPlayer(state.generateSuccessor(ghostCount, action), depth-1), action)
          return result

        def minHelper(v, maxResult, action):
          if maxResult[0] < v[0]:
            v[0] = maxResult[0]
            v[1] = action
          return v

        result = maxPlayer(gameState, depth)
        return result[1]

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
       depth = self.depth
       ghostCount = gameState.getNumAgents() - 1

       def maxPlayer(state, depth, alpha = float('-inf'), beta = float('inf')):
         if depth == 0:
           return [self.evaluationFunction(state), None]
         if state.getLegalActions(0) == []:
           return [self.evaluationFunction(state), None]
         v = [-float('inf'), None]
         for action in state.getLegalActions(0):
           result = maxHelper(v, minPlayer(state.generateSuccessor(0, action), 1, depth, alpha, beta), action)
           if result[0] > beta:
             return result
           alpha = max(alpha, result[0])
         return result
           
       def maxHelper(v, minResult, action):
         if minResult[0] > v[0]:
           v[0] = minResult[0]
           v[1] = action
         return v

       def minPlayer(state, ghostCount, depth, alpha = float('-inf'), beta = float('inf')):
         if depth == 0:
           return [self.evaluationFunction(state), None]
         if state.getLegalActions(ghostCount) == []:
           return [self.evaluationFunction(state), None]
         v = [float('inf'), None]
         for action in state.getLegalActions(ghostCount):
           if ghostCount != state.getNumAgents() - 1:
             result = minHelper(v, minPlayer(state.generateSuccessor(ghostCount, action), ghostCount + 1, depth, alpha, beta), action)
           else:
             result = minHelper(v, maxPlayer(state.generateSuccessor(ghostCount, action), depth-1, alpha, beta), action)
           if result[0] < alpha:
             return result
           beta = min(beta, result[0])

         return result

       def minHelper(v, maxResult, action):
         if maxResult[0] < v[0]:
           v[0] = maxResult[0]
           v[1] = action
         return v

       result = maxPlayer(gameState, depth)
       return result[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your ExpectiMax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the ExpectiMax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        ghostCount = gameState.getNumAgents() - 1

        def maxPlayer(state, depth):
            if depth == 0:
                return [self.evaluationFunction(state), None]
            if state.getLegalActions(0) == []:
                return [self.evaluationFunction(state), None]
            v = [-float('inf'), None]
            for action in state.getLegalActions(0):
                result = maxHelper(v, minPlayer(state.generateSuccessor(0, action), 1, depth), action)
            return result
            
        def maxHelper(v, minResult, action):
            if minResult[0] > v[0]:
                v[0] = minResult[0]
                v[1] = action
            return v

        def minPlayer(state, ghostCount, depth):
            if depth == 0:
                return [self.evaluationFunction(state), None]
            if state.getLegalActions(ghostCount) == []:
               return [self.evaluationFunction(state), None]
            exp = 0
            actions = state.getLegalActions(ghostCount)
            for action in actions:
                if ghostCount != state.getNumAgents() - 1:
                    result = minPlayer(state.generateSuccessor(ghostCount, action), ghostCount + 1, depth)
                    exp += result[0]
                else:
                    result = maxPlayer(state.generateSuccessor(ghostCount, action), depth-1)
                    exp += result[0]
                result[0] = float(exp)/float(len(actions))
            return result

        result = maxPlayer(gameState, depth)
        return result[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    def nearFood( pacPos, currGS, foodList):
      def helper(x, y):
        count = 0
        try:
          for a in range(x-2, x+3):
            for b in range(y-2, y+3):
             if currGS.hasFood(a, b) and not currGS.hasWall(a, b): count += 1
        except IndexError:
          pass
        return count
        #TODO: maybe handlee differently if count = small #
      return helper(pacPos[0], pacPos[1])


    def minGhostDist(pacPos, GameState):
      ghostPositions = GameState.getGhostPositions()
      minDist = 10000
      for ghostPosition in ghostPositions:
        d = manhattanDistance(pacPos, ghostPosition)
        if d < minDist and d != 0:
          minDist = d
      return minDist

    def minFoodDistance(pos, foodList):
      #calc min dist to a pellet
      minFoodDist = float("inf")
      foodList = foodList.asList()
      for f in foodList:
        d = manhattanDistance(pos, f)
        if d < minFoodDist:
          minFoodDist = d
      return minFoodDist

    #calc avg dist to a pellet
    def avgFoodDistance(pos, foodList):
      total, i = 0, 0
      foodList = foodList.asList()
      if len(foodList) != 1:
        for f in foodList:
          d = manhattanDistance(pos, f)
          i = i + 1
          # print(tot, ' ', i)
          total += d
        return total / (i+1)
      return 0.25


    pacPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    gameScore = currentGameState.getScore()
    foodTotal = currentGameState.getNumFood()
    foodList = currentGameState.getFood()

    #helpful features/stats
    foodCloseness = minFoodDistance(pacPos, currentFood)
    ghostCloseness = minGhostDist(pacPos, currentGameState)
    adjacentFoods = nearFood(pacPos, currentGameState, foodList)
    avgFoodDistance = avgFoodDistance(pacPos, currentFood)
    scared = [ghostState.scaredTimer for ghostState in currentGhostStates]
    scaredTime = scared[0]

    returnTotal = 0

    #basecase: no food ==> win
    if currentGameState.getNumFood() == 0 :
      returnTotal += 10001
    if currentGameState.getNumFood() == 1 :
      returnTotal += 5000

    if ghostCloseness < 3: #key to not getting eaten
      return ghostCloseness
    
    #If in empty zone due to avoiding ghost, move/slip by to new region
    if adjacentFoods == 0:
      returnTotal += 100/ghostCloseness 

    returnTotal = (20/foodCloseness) + (gameScore*20) - (100/(foodTotal+1)) \
     + adjacentFoods + 10/ghostCloseness + (40/(avgFoodDistance+1)) + (2*scaredTime)

    if debug: 
      print("returnTotal: ", returnTotal)

    return returnTotal

# Abbreviation
better = betterEvaluationFunction
