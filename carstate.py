from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from visualization import Visualization
from planarutils import SegmentCrossBox
from params import WorldParams, CarParams

######################################################################
#
#   State Definition
#
# Angular distance within +/- 180deg.
# TODO: Find a better place for this function
def AngleDiff(t1: float, t2: float) -> float:
    return (t1-t2) - 2.0*np.pi * round(0.5*(t1-t2)/np.pi)

#
#   State = One set of coordinates
#
class State:
    def __init__(self, x: float, y: float, theta: float, car: CarParams):
        # Pre-compute the trigonometry.
        s = np.sin(theta)
        c = np.cos(theta)

        # Remember the state (x,y,theta).
        self.x = x
        self.y = y
        self.t = theta
        self.s = s
        self.c = c

        self.wheelbase = car.wheelbase

        # Box (4 corners: frontleft, backleft, backright, frontright)
        self.box = ((x + c*car.lf - s*car.wc, y + s*car.lf + c*car.wc),
                    (x - c*car.lb - s*car.wc, y - s*car.lb + c*car.wc),
                    (x - c*car.lb + s*car.wc, y - s*car.lb - c*car.wc),
                    (x + c*car.lf + s*car.wc, y + s*car.lf - c*car.wc))

    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self) -> str:
        return ("<XY %5.2f,%5.2f @ %5.1f deg>" %
                (self.x, self.y, self.t * (180.0/np.pi)))

    # Draw the state.
    def Draw(self, fig: Visualization, color, **kwargs):
        b = self.box
        # Box
        plt.plot((b[0][0], b[1][0]), (b[0][1], b[1][1]), color, **kwargs)
        plt.plot((b[1][0], b[2][0]), (b[1][1], b[2][1]), color, **kwargs)
        plt.plot((b[2][0], b[3][0]), (b[2][1], b[3][1]), color, **kwargs)
        plt.plot((b[3][0], b[0][0]), (b[3][1], b[0][1]), color, **kwargs)
        # Headlights
        plt.plot(0.9*b[3][0]+0.1*b[0][0], 0.9*b[3][1]+0.1*b[0][1], color+'o')
        plt.plot(0.1*b[3][0]+0.9*b[0][0], 0.1*b[3][1]+0.9*b[0][1], color+'o')

    def DrawSimple(self, fig: Visualization, color, **kwargs):
        plt.plot(self.x, self.y, color+'o')
        plt.plot((self.x, self.x + self.c), 
                 (self.y, self.y + self.s), color, **kwargs)

    # Return a tuple of the coordinates for KDTree.
    def Coordinates(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.wheelbase*self.s, self.wheelbase*self.c)


    ############################################################
    # PRM Functions:
    # Check whether in free space.
    def InFreeSpace(self, world_params: WorldParams) -> bool:
        for wall in world_params.walls:
            if SegmentCrossBox(wall, self.box):
                return False
        return True

    # Compute the relative distance to another state.  Scale the
    # angular error by the car length.
    def Distance(self, other: State) -> float:
        return np.sqrt((self.x - other.x)**2 +
                       (self.y - other.y)**2 +
                       (self.wheelbase*AngleDiff(self.t, other.t))**2)

    def EuclideanDistance(self, other: State) -> float:
        return np.sqrt((self.x - other.x)**2 +
                       (self.y - other.y)**2)