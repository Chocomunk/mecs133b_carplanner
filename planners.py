from __future__ import annotations

import time
import bisect
import random
from abc import ABC, abstractclassmethod
from typing import List

import numpy as np
from sklearn.neighbors import KDTree

from carstate import State
from visualization import Visualization
from localplanners import LocalPlan
from params import WorldParams, CarParams

######################################################################
#
#   A* Functions
#
#
#   Node class upon which to build the graph (roadmap) and which
#   supports the A* search tree.
#
class Node:
    def __init__(self, state: State):
        # Save the state matching this node.
        self.state = state

        # Edges used for the graph structure (roadmap).
        self.childrenandcosts = []
        self.parents          = []

        # Status, edge, and costs for the A* search tree.
        self.seen        = False
        self.done        = False
        self.treeparent  = []
        self.costToReach = 0
        self.costToGoEst = np.inf
        self.cost        = self.costToReach + self.costToGoEst

    # Define the "less-than" to enable sorting by cost in A*.
    def __lt__(self, other: Node) -> bool:
        return self.cost < other.cost

    # Distance to another node, for A*, using the state distance.
    def Distance(self, other: Node) -> float:
        return self.state.Distance(other.state)


#
#   A* Planning Algorithm
#
def AStar(nodeList: List[Node], start: Node, goal: Node) -> List[Node]:
    # Prepare the still empty *sorted* on-deck queue.
    onDeck = []

    # Clear the search tree (for repeated searches).
    for node in nodeList:
        node.seen = False
        node.done = False

    # Begin with the start state on-deck.
    start.done        = False
    start.seen        = True
    start.treeparent  = None
    start.costToReach = 0
    start.costToGoEst = start.Distance(goal)
    start.cost        = start.costToReach + start.costToGoEst
    bisect.insort(onDeck, start)

    # Continually expand/build the search tree.
    while True:
        # Grab the next node (first on deck).
        node = onDeck.pop(0)

        # Add the children to the on-deck queue (or update)
        for (child,tripcost) in node.childrenandcosts:
            # Skip if already done.
            if child.done:
                continue

            # Compute the cost to reach the child via this new path.
            costToReach = node.costToReach + tripcost

            # Just add to on-deck if not yet seen (in correct order).
            if not child.seen:
                child.seen        = True
                child.treeparent  = node
                child.costToReach = costToReach
                child.costToGoEst = child.Distance(goal)
                child.cost        = child.costToReach + child.costToGoEst
                bisect.insort(onDeck, child)
                continue

            # Skip if the previous cost was better!
            if child.costToReach <= costToReach:
                continue

            # Update the child's connection and resort the on-deck queue.
            child.treeparent  = node
            child.costToReach = costToReach
            child.cost        = child.costToReach + child.costToGoEst
            onDeck.remove(child)
            bisect.insort(onDeck, child)

        # Declare this node done.
        node.done = True

        # Check whether we have processed the goal (now done).
        if (goal.done):
            break

        # Also make sure we still have something to look at!
        if not (len(onDeck) > 0):
            return []

    # Build the path.
    path = [goal]
    while path[0].treeparent is not None:
        path.insert(0, path[0].treeparent)

    # Return the path.
    return path


######################################################################
#
#   PRM Functions
#
#
# Sample the space
#

class Planner(ABC):
    
    @abstractclassmethod
    def search(self, startnode: Node, goalnode: Node, 
                visual: bool=False, fig: Visualization=None):
        pass


class PRMPlanner(Planner):

    def __init__(self, LocalPlanner: type[LocalPlan], world: WorldParams, car: CarParams, N: int, K: int):
        self.LocalPlanner = LocalPlanner
        self.world = world
        self.car = car
        self.N = N
        self.K = K

    def search(self, startnode: Node, goalnode: Node, 
                visual: bool=False, fig: Visualization=None):
        start_time = time.time()
        nodeList = []
        self.AddNodesToList(nodeList, self.N, startnode, goalnode)
        print('Sampling took ', time.time() - start_time)

        if visual and fig:
            # # Show the sample states.
            for node in nodeList:
                node.state.DrawSimple(fig, 'k', linewidth=1)
            fig.ShowFigure()
            input("Showing the nodes (hit return to continue)")

        # Add the start/goal nodes.
        nodeList.append(startnode)
        nodeList.append(goalnode)


        # Connect to the nearest neighbors.
        start_time = time.time()
        self.ConnectNearestNeighbors(nodeList, self.K)
        print('Connecting took ', time.time() - start_time)

        if visual and fig:
            # Show the neighbor connections.
            for node in nodeList:
                for (child, tripcost) in node.childrenandcosts:
                    plan = self.LocalPlanner(node.state, child.state, self.car)
                    plan.DrawSimple(fig, 'g-', linewidth=0.5)
            fig.ShowFigure()
            input("Showing the full graph (hit return to continue)")


        # Run the A* planner.
        start_time = time.time()
        path = AStar(nodeList, startnode, goalnode)
        print('A* took ', time.time() - start_time)
        return path

    def AddNodesToList(self, nodeList: List[Node], N: int, start: Node, goal: Node):
        xmin, xmax = self.world.xmin, self.world.xmax
        ymin, ymax = self.world.ymin, self.world.ymax

        # Add normally distributed samples around the start.
        mu_start = [start.state.x, start.state.y]
        # sig_start = [3, 2]
        sig_start = [(xmax - xmin) / 3, (ymax - ymin) / 3]
        while (len(nodeList) < N/2):
            # x, y = np.random.normal(mu_start, sig_start)
            # if xmin <= x <= xmax and ymin <= y <= ymax:
            #     t = random.uniform(-np.pi/4, np.pi/4)
            #     state = State(x,y,t,self.car)
            #     if state.InFreeSpace(self.world):
            #         nodeList.append(Node(state))
            state = State(random.uniform(xmin, xmax),
                        random.uniform(ymin, ymax),
                        random.uniform(-np.pi/4, np.pi/4),
                        self.car)
            if state.InFreeSpace(self.world):
                nodeList.append(Node(state))

        # Add normally distributed samples around the end.
        mu_goal = [goal.state.x, goal.state.y]
        sig_goal = [3, 2]
        # sig_goal = [(xmax - xmin) / 10, (ymax - ymin) / 10]
        while (len(nodeList) < N):
            x, y = np.random.normal(mu_goal, sig_goal)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                t = random.uniform(-np.pi/4, np.pi/4)
                state = State(x,y,t,self.car)
                if state.InFreeSpace(self.world):
                    nodeList.append(Node(state))


    #
    #   Connect the nearest neighbors
    #
    def ConnectNearestNeighbors(self, nodeList: List[Node], K: int):
        # Clear any existing neighbors.
        for node in nodeList:
            node.childrenandcosts = []
            node.parents          = []

        # Determine the indices for the nearest neighbors.  This also
        # reports the node itself as the closest neighbor, so add one
        # extra here and ignore the first element below.
        X   = np.array([node.state.Coordinates() for node in nodeList])
        kdt = KDTree(X)
        idx = kdt.query(X, k=(K+1), return_distance=False)

        # Add the edges (from parent to child).  Ignore the first neighbor
        # being itself.
        for i, nbrs in enumerate(idx):
            children = [child for (child,_) in nodeList[i].childrenandcosts]
            for n in nbrs[1:]:
                if not nodeList[n] in children:
                    plan = self.LocalPlanner(nodeList[i].state, nodeList[n].state, self.car)
                    if plan.Valid(self.world):
                        cost = plan.Length()
                        nodeList[i].childrenandcosts.append((nodeList[n], cost))
                        nodeList[n].childrenandcosts.append((nodeList[i], cost))
                        nodeList[n].parents.append(nodeList[i])
                        nodeList[i].parents.append(nodeList[n])
