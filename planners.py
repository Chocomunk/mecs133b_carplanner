from __future__ import annotations
from pathlib import Path

import time
import bisect
import random
from collections import deque
from abc import ABC, abstractclassmethod
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

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
    def __init__(self, state: State, parentnode: Node=None, draw: bool=False, color='r-'):
        # Save the state matching this node.
        self.state = state

        # Link to parent for RRT tree structure.
        self.parent = parentnode

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

        # Automatically draw.
        if draw:
            self.Draw(color, linewidth=1)

    # Draw a line to the parent.
    def Draw(self, *args, **kwargs):
        if self.parent is not None:
            plt.plot((self.state.x, self.parent.state.x),
                     (self.state.y, self.parent.state.y),
                     *args, **kwargs)
            plt.pause(0.001)

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
    def reset(self):
        pass
    
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

        self.nodeList = []

    def reset(self):
        del self.nodeList
        self.nodeList = []

    def search(self, startnode: Node, goalnode: Node, 
                visual: bool=False, fig: Visualization=None):
        # Build the graph 
        self.BuildGraph(startnode, goalnode, visual=visual, fig=fig)

        # Run the A* planner.
        start_time = time.time()
        path = AStar(self.nodeList, startnode, goalnode)
        print('A* took ', time.time() - start_time)
        return path

    def BuildGraph(self, startnode: Node, goalnode: Node, visual: bool=False, 
                    fig: Visualization=None, uniform: bool=False):
        start_time = time.time()
        self.AddNodesToList(self.nodeList, startnode, goalnode, uniform)
        print('Sampling took ', time.time() - start_time)

        if visual and fig:
            # # Show the sample states.
            for node in self.nodeList:
                node.state.DrawSimple(fig, 'k', linewidth=1)
            fig.ShowFigure()

        # Add the start/goal nodes.
        self.nodeList.append(startnode)
        self.nodeList.append(goalnode)


        # Connect to the nearest neighbors.
        start_time = time.time()
        self.ConnectNearestNeighbors(self.nodeList)
        print('Connecting took ', time.time() - start_time)

        if visual and fig:
            # Show the neighbor connections.
            for node in self.nodeList:
                for (child, tripcost) in node.childrenandcosts:
                    plan = self.LocalPlanner(node.state, child.state, self.car)
                    plan.DrawSimple(fig, 'g-', linewidth=0.5)
            fig.ShowFigure()


    def AddNodesToList(self, nodeList: List[Node], start: Node, goal: Node,
                        uniform: bool=False):
        xmin, xmax = self.world.xmin, self.world.xmax
        ymin, ymax = self.world.ymin, self.world.ymax
        
        if uniform:
            # Add uniformly distributed samples.
            while (len(nodeList) < self.N):
                state = State(random.uniform(xmin, xmax),
                            random.uniform(ymin, ymax),
                            random.uniform(-np.pi/4, np.pi/4),
                            self.car)
                if state.InFreeSpace(self.world):
                    nodeList.append(Node(state))
        else:
            # Add uniformly distributed samples.
            while (len(nodeList) < self.N/2):
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
            while (len(nodeList) < self.N):
                x, y = np.random.normal(mu_goal, sig_goal)
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    t = random.uniform(-np.pi/4, np.pi/4)
                    state = State(x,y,t,self.car)
                    if state.InFreeSpace(self.world):
                        nodeList.append(Node(state))

    def AddNodesToListUniform(self, nodeList: List[Node]):
        xmin, xmax = self.world.xmin, self.world.xmax
        ymin, ymax = self.world.ymin, self.world.ymax

        # Add uniformly distributed samples.
        while (len(nodeList) < self.N):
            state = State(random.uniform(xmin, xmax),
                        random.uniform(ymin, ymax),
                        random.uniform(-np.pi/4, np.pi/4),
                        self.car)
            if state.InFreeSpace(self.world):
                nodeList.append(Node(state))


    #
    #   Connect the nearest neighbors
    #
    def ConnectNearestNeighbors(self, nodeList: List[Node]):
        # Clear any existing neighbors.
        for node in nodeList:
            node.childrenandcosts = []
            node.parents          = []

        # Determine the indices for the nearest neighbors.  This also
        # reports the node itself as the closest neighbor, so add one
        # extra here and ignore the first element below.
        X   = np.array([node.state.Coordinates() for node in nodeList])
        kdt = KDTree(X)
        idx = kdt.query(X, k=(self.K+1), return_distance=False)

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


######################################################################
#
#  RRT Functions
#

class SpatialTable():

    def __init__(self, bin_xsize: float, bin_yize: float, init_factory: Callable,
                        xmin: float, xmax: float, ymin: float, ymax: float):
        """
            bin_xsize and bin_ysize are the dimensions of the individual bins.
            init_factory is a function that initializes each table entry.
        """
        self.xsize = bin_xsize
        self.ysize = bin_yize

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.width = xmax - xmin
        self.height = ymax - ymin
        
        self.xdim = int(np.ceil(self.width / self.xsize))
        self.ydim = int(np.ceil(self.height / self.ysize))

        self.table = [[init_factory() for _ in range(self.ydim)] for _ in range(self.xdim)]

    def coord_idx(self, x: float, y: float) -> Tuple[int, int]:
        return (int((x - self.xmin) // self.xsize), int((y - self.ymin) // self.ysize))

    def iget(self, x_ind: float, y_ind: float) -> Any:
        return self.table[x_ind][y_ind]

    def get(self, x: float, y: float) -> Any:
        x_ind, y_ind = self.coord_idx(x, y)
        return self.iget(x_ind, y_ind)

    def set(self, x: float, y: float, element: object):
        x_ind, y_ind = self.coord_idx(x, y)
        self.table[x_ind][y_ind] = element


def neighbors(x, y, dist, xmax, ymax):
    if x < 0 or x >= xmax or y < 0 or y >= ymax:
        return []
    out = []

    # Left wall
    x_shift = x - dist
    if x_shift >= 0:
        for y_shift in range(y-dist, y+dist+1):
            if 0 <= y_shift < ymax:
                out.append((x_shift, y_shift))

    # Right wall
    x_shift = x + dist
    if x_shift < xmax:
        for y_shift in range(y-dist, y+dist+1):
            if 0 <= y_shift < ymax:
                out.append((x_shift, y_shift))

    # Top wall
    y_shift = y - dist
    if y_shift >= 0:
        for x_shift in range(x-dist, x+dist+1):
            if 0 <= x_shift < ymax:
                out.append((x_shift, y_shift))

    # Bottom wall
    y_shift = y + dist
    if y_shift < ymax:
        for x_shift in range(x-dist, x+dist+1):
            if 0 <= x_shift < ymax:
                out.append((x_shift, y_shift))

    # Add current cell to closest dist
    if dist <= 1:
        out.append((x,y))

    return out

def nearest_node(target: State, bins: SpatialTable) -> Node:
    # This is inefficient (slow for large trees), but simple
    x, y = bins.coord_idx(target.x, target.y)
    xmax, ymax = bins.xdim, bins.ydim

    # Search radially outward in neighboring bins for nearest node.
    dist = 1
    nearnode = None
    min_dist = np.inf
    while not nearnode:
        nbrs = neighbors(x, y, dist, xmax, ymax)
        if not nbrs:
            return None
        for x_ind, y_ind in nbrs:
            for node in bins.iget(x_ind, y_ind):
                dist = node.state.Distance(target)
                if dist < min_dist:
                    min_dist = dist
                    nearnode = node
        dist += 1
    return nearnode

TARGET_SAMPLES = 5
class RRTPlanner(Planner):

    def __init__(self, LocalPlanner: type[LocalPlan], world: WorldParams, 
                car: CarParams, Nmax: int, dstep: float, color='r-'):
        self.LocalPlanner = LocalPlanner
        self.world = world
        self.car = car
        self.Nmax = Nmax
        self.dstep = dstep
        self.color = color

        # NOTE: Check 'reset()' if params are changed
        self.node_count = 0
        self.node_bins = SpatialTable(5, 5, list,
                                    self.world.xmin, self.world.xmax, 
                                    self.world.ymin, self.world.ymax)
        self.sample_bins = SpatialTable(5, 5, int,
                                    self.world.xmin, self.world.xmax, 
                                    self.world.ymin, self.world.ymax)

    # Generate sample from uniform distribution
    def sample_target(self, goalstate: State) -> State:
        if random.random() < 0.05:
            return goalstate

        # Draw 5 samples
        samples = []
        for _ in range(TARGET_SAMPLES):
            x = random.uniform(self.world.xmin, self.world.xmax)
            y = random.uniform(self.world.ymin, self.world.ymax)
            t = random.uniform(-np.pi/4, np.pi/4)
            freq = len(self.node_bins.get(x, y)) + self.sample_bins.get(x, y)
            samples.append((freq, (x, y, t)))

        # Select lowest density sample
        _, (x, y, t) = min(samples, key=lambda x: x[0])

        # Update sample density
        for _, (x, y, t) in samples:
            x_ind, y_ind = self.sample_bins.coord_idx(x, y)
            self.sample_bins.table[x_ind][y_ind] += 1

        return State(x, y, t, self.car)

    def add_node(self, node: Node, parent: Optional[Node]):
        node.parent = parent
        self.node_count += 1
        self.node_bins.get(node.state.x, node.state.y).append(node)

    def reset(self):
        del self.node_bins
        del self.sample_bins

        self.node_count = 0
        self.node_bins = SpatialTable(5, 5, list,
                                    self.world.xmin, self.world.xmax, 
                                    self.world.ymin, self.world.ymax)
        self.sample_bins = SpatialTable(5, 5, int,
                                    self.world.xmin, self.world.xmax, 
                                    self.world.ymin, self.world.ymax)

    def grow(self, targetstate: State, visual=False) -> Optional[Node]:
        # Find the nearest node (node with state nearest the target state).
        nearnode = nearest_node(targetstate, self.node_bins)
        nearstate = nearnode.state

        # Determine the next state, a step size (dstep) away.
        plan = self.LocalPlanner(nearstate, targetstate, self.car)
        nextstate = plan.IntermediateState(self.dstep * plan.Length())

        # Don't add if another node is already there
        closest = nearest_node(nextstate, self.node_bins)
        if closest and nextstate.Distance(closest.state) < 0.01:
            return None

        # Check whether to attach (creating a new node).
        if self.LocalPlanner(nearstate, nextstate, self.car).Valid(self.world):
            nextnode = Node(nextstate, nearnode, draw=visual, color=self.color)
            self.add_node(nextnode, nearnode)
            return nextnode
        return None
    
    def search(self, startnode: Node, goalnode: Node, visual=False, fig: Visualization=None):
        # Start the tree with the start state and no parent.
        self.add_node(startnode, None)

        time_avg = 0
        while self.node_count < self.Nmax:
            start_time = time.time()

            # Determine the target state.
            targetstate = self.sample_target(goalnode.state)
            nextnode = self.grow(targetstate, visual=visual)

            # Print average single-node growth time.
            time_avg += time.time() - start_time
            if self.node_count % 100 == 0:
                print("{0}/{1} Average growth time: {2:0.5f} s".format(
                    self.node_count, self.Nmax, time_avg / 100))
                time_avg = 0
            
            # If next node found, also try to connect the goal.
            if nextnode:
                goal_plan = self.LocalPlanner(nextnode.state, goalnode.state, self.car)
                if goal_plan.Valid(self.world):
                    self.add_node(goalnode, nextnode)

                    # Construct path and return
                    path = [goalnode]
                    while path[-1].parent is not None:
                        path.append(path[-1].parent)
                    return reversed(path)

        return None


class RRT2TreePlanner(Planner):

    def __init__(self, LocalPlanner: type[LocalPlan], world: WorldParams, car: CarParams, Nmax: int, dstep: float):
        self.LocalPlanner = LocalPlanner
        self.world = world
        self.car = car
        self.Nmax = Nmax
        self.dstep = dstep

        self.node_count = 0
        self.tree1 = RRTPlanner(LocalPlanner, world, car, Nmax, dstep, color='r-')
        self.tree2 = RRTPlanner(LocalPlanner, world, car, Nmax, dstep, color='g-')

    def reset(self):
        self.node_count = 0
        self.tree1.reset()
        self.tree2.reset()

    def search(self, startnode: Node, goalnode: Node, visual=False, fig: Visualization=None):
        # Start the tree with the start state and no parent.
        self.tree1.add_node(startnode, None)
        self.tree2.add_node(goalnode, None)

        T1 = self.tree1
        T2 = self.tree2

        while self.node_count < self.Nmax:
            # Determine the target state.
            targetstate = T1.sample_target(goalnode.state)
            new1 = T1.grow(targetstate, visual=visual)
            
            # If next node found, also try to connect the goal.
            if new1:
                new2 = T2.grow(new1.state, visual=visual)
                if new2:
                    goal_plan = self.LocalPlanner(new1.state, new2.state, self.car)
                    if goal_plan.Valid(self.world):
                        # Construct path and return
                        path = [new1]
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)
                        path = list(reversed(path))
                        path.append(new2)
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)

                        return path
            
                # Swap trees
                T1, T2 = T2, T1

        return None


class PRM2TreePlanner(Planner):

    def __init__(self, LocalPlanner: type[LocalPlan], world: WorldParams, car: CarParams, 
                Nmax: int, dstep: float, prmN: int, prmK: int):
        self.LocalPlanner = LocalPlanner
        self.world = world
        self.car = car
        self.Nmax = Nmax
        self.dstep = dstep
        self.N = prmN
        self.K = prmK

        self.node_count = 0
        self.prm = PRMPlanner(LocalPlanner, world, car, prmN, prmK)
        self.tree1 = RRTPlanner(LocalPlanner, world, car, Nmax, dstep, color='r-')
        self.tree2 = RRTPlanner(LocalPlanner, world, car, Nmax, dstep, color='m-')
        self.prm_samples = SpatialTable(5, 5, list,
                                self.world.xmin, self.world.xmax, 
                                self.world.ymin, self.world.ymax)

    def reset(self):
        del self.prm_samples

        self.node_count = 0
        self.prm_samples = SpatialTable(5, 5, list,
                                self.world.xmin, self.world.xmax, 
                                self.world.ymin, self.world.ymax)
        self.tree1.reset()
        self.tree2.reset()

    def connect_prm(self, tree: RRTPlanner, node: Node):
        newnode = nearest_node(node.state, self.prm_samples)
        if newnode:
            plan = self.LocalPlanner(node.state, newnode.state, self.car)
            if plan.Valid(self.world):
                tree.add_node(newnode, node)
                newnode.Draw(tree.color, linewidth=1)
                self.hook_graph(tree, newnode)

    def hook_graph(self, tree: RRTPlanner, root: Node):
        q = deque([root])
        while q:
            curr = q.popleft()
            self.prm_samples.get(curr.state.x, curr.state.y).remove(curr)
            for node, _ in curr.childrenandcosts:
                if not node.parent:     # If RRT parent has not been claimed
                    tree.add_node(node, curr)
                    if curr.parent == node:
                        print(curr.childrenandcosts, node.childrenandcosts)
                    node.Draw(tree.color, linewidth=1)
                    q.append(node)

    def search(self, startnode: Node, goalnode: Node, visual=False, fig: Visualization=None):
        # Generate a preliminary PRM graph
        self.prm.BuildGraph(startnode, goalnode, False, fig, uniform=True)
        for node in self.prm.nodeList:
            self.prm_samples.get(node.state.x, node.state.y).append(node)

        # Start the tree with the start state and no parent.
        self.tree1.add_node(startnode, None)
        self.tree2.add_node(goalnode, None)

        T1 = self.tree1
        T2 = self.tree2

        while self.node_count < self.Nmax:
            # Determine the target state.
            targetstate = T1.sample_target(goalnode.state)
            new1 = T1.grow(targetstate, visual=visual)
            
            # If next node found, also try to connect the goal.
            if new1:
                new2 = T2.grow(new1.state, visual=visual)
                if new2:
                    goal_plan = self.LocalPlanner(new1.state, new2.state, self.car)
                    if goal_plan.Valid(self.world):
                        # Construct path and return
                        path = [new1]
                        curr = new1
                        while curr.parent is not None:
                            print(path[-1], path[-1].parent, path[-1].parents, path[-1].childrenandcosts)
                            path.append(curr.parent)
                            curr = curr.parent
                        path = list(reversed(path))
                        path.append(new2)
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)

                        return path
                    self.connect_prm(T2, new2)
                self.connect_prm(T1, new1)
            
                # Swap trees
                T1, T2 = T2, T1

        return None
