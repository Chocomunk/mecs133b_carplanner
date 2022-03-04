#!/usr/bin/env python3
#
#   prmcar.py
#

import os
from pickle import FALSE
import sys
import numpy as np
import time

from carstate import State
from planners import Node, Planner, PRMPlanner, RRTPlanner, RRT2TreePlanner
from visualization import Visualization
from pathprocessing import PathProcessor
from params import WorldParams, ZigZagWorld, WallWorld, BlockWorld, SmallWorld, CarParams
from localplanners import LocalPlan, LocalPlan2Arc, LocalPlan3Arc, LocalPlan4Arc


# PRM parameters
N = 200
K = 40
Nmax = 2000
dstep = 0.1


# Flags
TRIALS = 1                # Only 1 trial is run if SHOW_VISUAL is True
RUN_TESTS = False
SHOW_VISUAL = True
LOOP_TO_SUCCESS = True
CHECK_PATH_STEPS = False


######################################################################
#
#  Main Code
#
def CheckLocalPlan(fig: Visualization, LocalPlanner: type[LocalPlan], 
                    fromState: State, toState: State, c: CarParams):
    # Clear the figure.
    fig.ClearFigure()

    # Show the initial and final states.
    fromState.Draw(fig, 'r', linewidth=2)
    toState.Draw(fig,   'r', linewidth=2)

    # Compute and show the local plan.
    plan = LocalPlanner(fromState, toState, c)
    plan.Draw(fig, 'b', linewidth=1)

    # Show/report
    fig.ShowFigure()
    print("Local plan from %s to %s" % (fromState, toState))
    input("(hit return to continue)")

def TestLocalPlanner(fig: Visualization, LocalPlanner: type[LocalPlan], c: CarParams):
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(8,  0, 0, c))
    CheckLocalPlan(fig, LocalPlanner, State(8, 0, 0, c),       State(0,  0, 0, c))
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(8,  8, np.pi/2, c))
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(8,  8, 0, c))
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(8, -8, 0, c))
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(8,  0, -np.pi/4, c))

    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(0,  8, 0, c))
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(0,  8, np.pi/2, c))

    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(6, -6,   np.pi/2, c))
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(8, -8,   np.pi/2, c))
    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(10, -10,   np.pi/2, c))

    CheckLocalPlan(fig, LocalPlanner, State(0, 0, 0, c),       State(7, 0,   np.pi, c))
    CheckLocalPlan(fig, LocalPlanner, State(-10, 0, 0, c),     State(10, 0,   np.pi, c))

    
def main() -> bool:
    # Report the parameters.
    print('Running with ', N, ' nodes and ', K, ' neighbors.')

    # Create the managers
    if SHOW_VISUAL:
        fig = Visualization()
    c = CarParams()
    wp = WallWorld()
    LocalPlanner: type[LocalPlan] = LocalPlan3Arc
    # planner: Planner = PRMPlanner(LocalPlanner, wp, c, N, K)
    planner: Planner = RRT2TreePlanner(LocalPlanner, wp, c, Nmax, dstep)
    path_processor = PathProcessor(LocalPlanner, wp, c)

    # Test the local planner:
    if RUN_TESTS:
        TestLocalPlanner(fig, LocalPlanner, c)

    if SHOW_VISUAL:
        # Switch to the road figure.
        fig.ClearFigure()
        fig.ShowParkingSpot(wp)

    # Pick your start and goal locations.
    (startx, starty, startt) = (2.0, 2.0, 0.0)
    (goalx,  goaly,  goalt)  = (
        wp.xspace + (wp.lspace-c.lcar)/2 + c.lb, 
        wp.wroad + c.wc, 
        0.0)
        
    # Create the start/goal nodes.
    startnode = Node(State(startx, starty, startt, c))
    goalnode  = Node(State(goalx,  goaly,  goalt, c))

    if SHOW_VISUAL:
        # Show the start/goal states.
        startnode.state.Draw(fig, 'r', linewidth=2)
        goalnode.state.Draw(fig,  'r', linewidth=2)
        fig.ShowFigure()

    # Create the list of sample points.
    path = planner.search(startnode, goalnode, visual=True, fig=fig)
    if not path:
        print("UNABLE TO FIND A PATH")
        return False

    if SHOW_VISUAL:
        # Show the path.
        path_processor.DrawPath(path, fig, 'r', CHECK_PATH_STEPS, linewidth=1)
        fig.ShowFigure()
        input("Showing the raw path (hit return to continue)")

        # Post Process the path.
        path = path_processor.PostProcess(path)

        # Show the post-processed path.
        path_processor.DrawPath(path, fig, 'b', CHECK_PATH_STEPS, linewidth=2)
        fig.ShowFigure()
        input("Showing the post-processed path (hit return to continue)")

    return True


# Disable prints
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore prints
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__== "__main__":
    if SHOW_VISUAL:
        ret = False
        while not ret:
            ret = main() or not LOOP_TO_SUCCESS

    else:
        print("Running {0} trials for N={1} and K={2}".format(TRIALS, N, K))

        start_time = time.time()
        blockPrint()
        results = np.array([main() for _ in range(TRIALS)])
        enablePrint()
        print("Finished running in {0:.4f} seconds".format(time.time() - start_time))

        success_rate = np.mean(results)

        print("Success Rate: {0:.4f}".format(success_rate))
        print("Took {0} tries".format(len(results)))
