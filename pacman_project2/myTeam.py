# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Directions
import game
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffenseMyTeam', second = 'DefenseMyTeam'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class BaseMyTeam(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.prev = Directions.STOP
    self.capsules = -1
    self.counter = -1
    #self.ind = 0

    self.totalEnemyFood = self.getFood(gameState).asList()

    if self.index % 2 == 0:
      boundary = (gameState.data.layout.width - 2) / 2
    else:
      boundary = ((gameState.data.layout.width - 2) / 2) + 1
    boundary = math.floor(boundary)
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(boundary, i):
        self.boundary.append((boundary, i))
    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    #print(actions)
    if len(actions) <= 2:
      pass

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


class OffenseMyTeam(BaseMyTeam):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    #if(self.ind == 1):
    print (values)
    print (actions)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    #if self.counter > 0:
      #self.counter = self.counter - 1

    #if(self.capsules > len(self.getCapsules(gameState))):
      #self.counter = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    myPos = gameState.getAgentState(self.index).getPosition()
    dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
    self.counter = min([enemy.scaredTimer for enemy in defenders if min(dists) == self.getMazeDistance(myPos, enemy.getPosition())])
    print ("Counter: ",self.counter)
    self.capsules = len(self.getCapsules(gameState))
    if self.prev in bestActions and self.ind == 1:
      print ("ponavlja najbolji")
      print (bestActions)

    #if foodLeft <= 2:

      #bestDist = 9999
      #for action in actions:
        #successor = self.getSuccessor(gameState, action)
        #pos2 = successor.getAgentPosition(self.index)
        #dist = self.getMazeDistance(self.start, pos2)
        #if dist < bestDist:
          #bestAction = action
          #bestDist = dist
      #self.prev = bestAction
      #return bestAction

    self.prev = random.choice(bestActions)

    return self.prev

  def depthAction(self, gameState, currDist, agentPos, depth):
    actions = gameState.getLegalActions(self.index)
    if len(actions) <= 2:
      #print(depth)
      #print(actions)
      return 0;
    if depth == 0:

      for action in actions:
        if action == Directions.STOP:
          continue
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        pomval = -9999
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(defenders) > 0:
          dists = [self.getMazeDistance(pos2, a.getPosition()) for a in defenders]
          pomval = min(dists)
        if currDist <= pomval and agentPos != pos2:
          return 1000
    else:
      for action in actions:
        if action == Directions.STOP:
          continue
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        pomval = -9999
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(defenders) > 0:
          dists = [self.getMazeDistance(pos2, a.getPosition()) for a in defenders]
          pomval = min(dists)

        dist = self.getMazeDistance(self.start, pos2)
        pomState = gameState.deepCopy()

        nextActionVal = self.depthAction(pomState.generateSuccessor(self.index, action), currDist, agentPos, depth - 1)
        if nextActionVal > 0:
          return  nextActionVal
    return 0


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    capsuleList = self.getCapsules(successor)
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    myPos = successor.getAgentState(self.index).getPosition()
    minBoundaryDist = min([self.getMazeDistance(myPos, boundary) for boundary in self.boundary])
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    scaredValue = 5
    responseValue = 7
    #print (self.numCarr)
    if len(capsuleList) == 0 and successor.getAgentState(self.index).numCarrying <= 5 and self.getScore(gameState) > 0:
      responseValue = 2
      scaredValue = 2
    if myPos == self.start:
      features['eaten'] = -1
    if action == Directions.STOP: features['stop'] = 1
    self.ind = 0
    if (len(foodList) < len(self.totalEnemyFood) and len(capsuleList) <= 0 and self.getScore(gameState) <= 0 and minBoundaryDist <= 5) or len(foodList) <= 2:
      print ("usao")

      features['distanceBoundary'] =  minBoundaryDist
      features['distanceToFood'] = 0
      features['successorScore'] = 0
      if len(defenders) > 0 and self.counter <= 3:
        dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
        features['distanceToGhost'] =  min(dists)
        if min(dists) > minBoundaryDist:
          features['distanceBoundary'] = 5 * minBoundaryDist
          features['distanceToGhost'] = 10000
          print("bezim")
          print (action)
          print (features)
        else:
          if min([self.getMazeDistance(boundary, a.getPosition()) for a in defenders for boundary in self.boundary]) > minBoundaryDist:
            features['distanceBoundary'] = 5 * minBoundaryDist
            features['distanceToGhost'] = 10000
            print (action)
            print (features)
      return features

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    if action == self.prev:
      features['repeatAction'] = -1

    if len(defenders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
      realPos = gameState.getAgentState(self.index).getPosition()
      realDist = [self.getMazeDistance(realPos, a.getPosition()) for a in defenders]
      #if self.counter > 20:
        #if (len(capsuleList) > 0):
          #minCapsuleDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
          #features['distanceToCapsule'] = -minCapsuleDist
      if self.counter <= 3:
        if min(realDist) <= responseValue:
          if Directions.REVERSE[action] == self.prev:
            features['repeatAction'] = -10
          features['distanceToGhost'] = min(dists) #(-2000 succScore retko !=) -15 distFood x * distEnemie
          #print (action)
          #print (dists)
          self.ind = 1
          features['distanceBoundary'] = minBoundaryDist
          if(len(capsuleList) > 0):
            minCapsuleDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            features['distanceToCapsule'] = minCapsuleDist
            #print (action)
            #print (minCapsuleDist)
          else:
            features['distanceBoundary'] = minBoundaryDist
            features['distanceToGhost'] = min(dists)
            features['distanceToFood'] = 0
            features['successorScore'] = 0
            print (action)
            print (features)
            return features
          features['eatCapsule'] = -len(capsuleList)
          #print (minCapsuleDist)

          #print (features)
          #print (action)

        if min(realDist) <= scaredValue:
          features['distanceToFood'] = 0
          features['successorScore'] = -5000

      elif self.counter <= 10:
        #print (action)
        #print (features)
        if min(realDist) < self.counter - 3:
          features["distanceToGhost"] = min(dists) * -500
          if min(dists) == 0:
            features["distanceToGhost"] = 1000
      #print (min(dists))
    print (features)
    print (action)

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1000, 'distanceToFood': -0.5, "distanceToGhost": 5, "stop": -99999, 'repeatAction': 0.23,
            'distanceToCapsule': -4, 'eatCapsule': 1000, 'distanceBoundary': -2, 'eaten': 500000}


class DefenseMyTeam(BaseMyTeam):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}