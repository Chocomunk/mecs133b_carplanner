from __future__ import annotations

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
#   Utility Definitions
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
#   SpatialTable
#       divides the world into 'bins' which store data that are in the same 
#       local area.
#
class SpatialTable():
    """ divides the world into 'bins' which store data that are in the same 
        local area.
    """

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
        """ Convert a coordinate to a bin index """
        return (int((x - self.xmin) // self.xsize), int((y - self.ymin) // self.ysize))

    def iget(self, x_ind: float, y_ind: float) -> Any:
        """ Index from internal table by pure index """
        return self.table[x_ind][y_ind]

    def get(self, x: float, y: float) -> Any:
        """ Index from internal table by coordinate """
        x_ind, y_ind = self.coord_idx(x, y)
        return self.iget(x_ind, y_ind)


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


def neighbors(x, y, dist, xmax, ymax):
    """ Return a list of valid coordinates that are on a square of radius `dist`
        from `(x,y)`.
    """
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

def nearest_node(target: State, bins: SpatialTable, dist_lim: int=0) -> Node:
    """ Finds the node in `bins` that is closest to `target` """
    x, y = bins.coord_idx(target.x, target.y)
    xmax, ymax = bins.xdim, bins.ydim

    # Search radially outward in neighboring bins for nearest node.
    dist = 1
    nearnode = None
    min_dist = np.inf
    while not nearnode and (dist_lim <= 0 or dist < dist_lim):
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


######################################################################
#
#   Planner (Abstract Class)
#
#
# The base class for all planners.
#

class Planner(ABC):

    @abstractclassmethod
    def reset(self):
        """ Resets the internal data structures for this planner """
        pass
    
    @abstractclassmethod
    def search(self, startnode: Node, goalnode: Node, 
                visual: bool=False, fig: Visualization=None):
        """ Returns a path of nodes from `startnode` to `goalnode` """
        pass


######################################################################
#
#   PRM Planner
#
#
# Sample the space
#

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
        start_time = time.time()
        self.sample_improved(goalnode)
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
        self.ConnectNearestNeighbors()
        print('Connecting took ', time.time() - start_time)

        if visual and fig:
            # Show the neighbor connections.
            for node in self.nodeList:
                for (child, tripcost) in node.childrenandcosts:
                    plan = self.LocalPlanner(node.state, child.state, self.car)
                    plan.DrawSimple(fig, 'g-', linewidth=0.5)
            fig.ShowFigure()

        # Run the A* planner.
        start_time = time.time()
        path = AStar(self.nodeList, startnode, goalnode)
        print('A* took ', time.time() - start_time)
        return path

    def sample_improved(self, goal: Node):
        xmin, xmax = self.world.xmin, self.world.xmax
        ymin, ymax = self.world.ymin, self.world.ymax
        
        # Add uniformly distributed samples.
        while (len(self.nodeList) < self.N/2):
            state = State(random.uniform(xmin, xmax),
                        random.uniform(ymin, ymax),
                        random.uniform(-np.pi/4, np.pi/4),
                        self.car)
            if state.InFreeSpace(self.world):
                self.nodeList.append(Node(state))

        # Add normally distributed samples around the end.
        mu_goal = [goal.state.x, goal.state.y]
        sig_goal = [3, 2]
        while (len(self.nodeList) < self.N):
            x, y = np.random.normal(mu_goal, sig_goal)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                t = random.uniform(-np.pi/4, np.pi/4)
                state = State(x,y,t,self.car)
                if state.InFreeSpace(self.world):
                    self.nodeList.append(Node(state))

    def sample_uniform(self):
        xmin, xmax = self.world.xmin, self.world.xmax
        ymin, ymax = self.world.ymin, self.world.ymax

        # Add uniformly distributed samples.
        while (len(self.nodeList) < self.N):
            state = State(random.uniform(xmin, xmax),
                        random.uniform(ymin, ymax),
                        random.uniform(-np.pi/4, np.pi/4),
                        self.car)
            if state.InFreeSpace(self.world):
                self.nodeList.append(Node(state))

    def ConnectNearestNeighbors(self):
        """ Connect nearest neighbor nodes into a graph """
        # Clear any existing neighbors.
        for node in self.nodeList:
            node.childrenandcosts = []
            node.parents          = []

        # Determine the indices for the nearest neighbors.  This also
        # reports the node itself as the closest neighbor, so add one
        # extra here and ignore the first element below.
        X   = np.array([node.state.Coordinates() for node in self.nodeList])
        kdt = KDTree(X)
        idx = kdt.query(X, k=(self.K+1), return_distance=False)

        # Add the edges (from parent to child).  Ignore the first neighbor
        # being itself.
        for i, nbrs in enumerate(idx):
            children = [child for (child,_) in self.nodeList[i].childrenandcosts]
            for n in nbrs[1:]:
                if not self.nodeList[n] in children:
                    plan = self.LocalPlanner(self.nodeList[i].state, self.nodeList[n].state, self.car)
                    if plan.Valid(self.world):
                        cost = plan.Length()
                        self.nodeList[i].childrenandcosts.append((self.nodeList[n], cost))
                        self.nodeList[n].childrenandcosts.append((self.nodeList[i], cost))
                        self.nodeList[n].parents.append(self.nodeList[i])
                        self.nodeList[i].parents.append(self.nodeList[n])


TARGET_SAMPLES = 5


######################################################################
#
#  RRT Planners (including 2-tree)
#

class RRTPlanner(Planner):

    def __init__(self, LocalPlanner: type[LocalPlan], world: WorldParams, 
                car: CarParams, Nmax: int, dstep: float, color='r-'):
        self.LocalPlanner = LocalPlanner
        self.world = world
        self.car = car
        self.Nmax = Nmax
        self.dstep = dstep
        self.color = color

        self.node_count = 0
        self.node_bins = SpatialTable(5, 5, list,
                                    self.world.xmin, self.world.xmax, 
                                    self.world.ymin, self.world.ymax)
        self.sample_bins = SpatialTable(5, 5, int,
                                    self.world.xmin, self.world.xmax, 
                                    self.world.ymin, self.world.ymax)

    def sample_target(self, goalstate: State) -> State:
        """ Generate a random target. 
            Uses the spatial tables to favor less-dense areas of the world.
         """
        if random.random() < 0.05:
            return goalstate

        # Draw 5 samples
        samples = []
        for _ in range(TARGET_SAMPLES):
            x = random.uniform(self.world.xmin, self.world.xmax)
            y = random.uniform(self.world.ymin, self.world.ymax)
            t = random.uniform(-np.pi/4, np.pi/4)

            # Use spatial tables to approximate density around the sample
            # density = {# nodes added in area} + {# samples taken in area}
            density = len(self.node_bins.get(x, y)) + self.sample_bins.get(x, y)
            samples.append((density, (x, y, t)))

        # Select lowest density sample
        _, (x, y, t) = min(samples, key=lambda x: x[0])

        # Update sample density
        for _, (x, y, t) in samples:
            x_ind, y_ind = self.sample_bins.coord_idx(x, y)
            self.sample_bins.table[x_ind][y_ind] += 1

        return State(x, y, t, self.car)

    def add_node(self, node: Node, parent: Optional[Node]):
        """ Attaches `node` to `parent` in the tree. Updates the density table. """
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
        """ Attempt to add a new node that grows towards `targetstate` """
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
            nextnode = Node(nextstate)
            self.add_node(nextnode, nearnode)
            if visual:
                nextnode.Draw(self.color, linewidth=1)
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
            
            # If next node found, also try to connect to T2.
            if new1:
                new2 = T2.grow(new1.state, visual=visual)
                if new2:
                    connected = self.LocalPlanner(new1.state, new2.state, self.car)
                    if connected.Valid(self.world):
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


######################################################################
#
#   Hybrid PRM/2-tree planner
#
#
# Sample a sparse graph that the RRT tree can attach to/jump through.
#

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

    def connect_prm(self, tree: RRTPlanner, treenode: Node):
        """ Search for a PRM graph unit to connect with `tree` """
        prmnode = nearest_node(treenode.state, self.prm_samples, dist_lim=2)
        if prmnode and not prmnode.parent:
            plan = self.LocalPlanner(treenode.state, prmnode.state, self.car)
            if plan.Valid(self.world):
                tree.add_node(prmnode, treenode)
                prmnode.Draw(tree.color, linewidth=1)
                self.hook_graph(tree, prmnode)

    def hook_graph(self, tree: RRTPlanner, root: Node):
        """ Do a BFS to attach a graph to `tree` from `root` in the graph """
        q = deque([root])
        while q:
            curr = q.popleft()
            self.prm_samples.get(curr.state.x, curr.state.y).remove(curr)
            for child, _ in curr.childrenandcosts:
                # If RRT parent has not been claimed and child is available
                if not child.parent:     
                    tree.add_node(child, curr)
                    # child.Draw(tree.color, linewidth=1)
                    q.append(child)

    def search(self, startnode: Node, goalnode: Node, visual=False, fig: Visualization=None):
        # Generate a preliminary PRM graph (don't include start/goal)
        self.prm.sample_uniform()
        self.prm.ConnectNearestNeighbors()
        for prmnode in self.prm.nodeList:
            self.prm_samples.get(prmnode.state.x, prmnode.state.y).append(prmnode)

        if visual and fig:
            # Show the neighbor connections.
            for node in self.prm.nodeList:
                node.state.DrawSimple(fig, 'k', linewidth=1)
                for (child, tripcost) in node.childrenandcosts:
                    plan = self.LocalPlanner(node.state, child.state, self.car)
                    plan.DrawSimple(fig, 'g-', linewidth=0.5)
            fig.ShowFigure()

        # Start the tree with the start state and no parent.
        self.tree1.add_node(startnode, None)
        self.tree2.add_node(goalnode, None)

        T1 = self.tree1
        T2 = self.tree2

        while self.node_count < self.Nmax:
            # Determine the target state.
            targetstate = T1.sample_target(goalnode.state)
            new1 = T1.grow(targetstate, visual=visual)
            
            # If next node found, also try to connect to T2.
            if new1:
                new2 = T2.grow(new1.state, visual=visual)
                if new2:
                    connected = self.LocalPlanner(new1.state, new2.state, self.car)
                    if connected.Valid(self.world):
                        # Construct path and return
                        path = [new1]
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)
                        path = list(reversed(path))
                        path.append(new2)
                        while path[-1].parent is not None:
                            path.append(path[-1].parent)

                        return path

                # Attempt to connect to PRM graph units
                    self.connect_prm(T2, new2)
                self.connect_prm(T1, new1)
            
                # Swap trees
                T1, T2 = T2, T1

        return None
