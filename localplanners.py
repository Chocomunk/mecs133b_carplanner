from abc import ABC, abstractclassmethod
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from carstate import State
from params import WorldParams, CarParams
from planarutils import SegmentCrossArc, SegmentCrossSegment, AngleDiff
from visualization import Visualization, DrawParams


######################################################################
#
#   Local Planner
#
#   There are many options here.  We assume we drive at a constant
#   speed and for a given distance at one steering angle (turning
#   radius), followed by another distance at the opposite steering
#   angle.  As such, we also define an arc, being a constant speed and
#   steering angle.
#
#   Note the tan(steeringAngle) = wheelBase / turningRadius
#
class Arc:
    def __init__(self, fromState: State, toState: State, distance: float, radius: float):
        # Remember the parameters.
        self.fromState = fromState
        self.toState   = toState
        self.distance  = distance       # can be negative when backing up!
        self.r         = radius         # pos = turn left, neg = turn right

    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self) -> str:
        return ("<Arc %s to %s, distance %5.2f m, radius %5.2f m>" %
                (self.fromState, self.toState, self.distance, self.r))

    # Return the absolute length.
    def Length(self) -> float:
        return abs(self.distance)

    # Return an intermediate state, d along arc.  Note we take care
    # never to divide, so this works for both arcs and straight lines.
    def IntermediateState(self, d: float, car: CarParams) -> State:
        if self.r is None:
            return State(self.fromState.x + self.fromState.c * d,
                         self.fromState.y + self.fromState.s * d,
                         self.fromState.t, 
                         car)
        else:
            phi = d / self.r
            ds  = np.sin(self.fromState.t + phi) - self.fromState.s
            dc  = np.cos(self.fromState.t + phi) - self.fromState.c
            return State(self.fromState.x + self.r * ds,
                         self.fromState.y - self.r * dc,
                         self.fromState.t + phi,
                         car)

    # Draw the arc (showing the intermediate states).
    def Draw(self, car: CarParams, fig: Visualization, color, **kwargs):
        n = max(2, int(np.ceil(abs(self.distance) / DrawParams.ddraw)))
        d = self.distance / n
        for i in range(1,n):
            self.IntermediateState(d*i, car).Draw(fig, color, **kwargs)

    ############################################################
    # PRM Functions:
    # Check whether in free space, with given spacing.
    def Valid(self, world_params: WorldParams, rmin: float) -> bool:
        # First, check if straight: Check back left/right corners segments.
        if self.r is None:
            seg1 = (self.fromState.box[1], self.toState.box[1])
            seg2 = (self.fromState.box[2], self.toState.box[2])
            for wall in world_params.walls:
                if (SegmentCrossSegment(wall, seg1) or
                    SegmentCrossSegment(wall, seg2)):
                    return False
            return True

        # If not straight, then check the min turning radius.
        if abs(self.r) < rmin:
            return False

        # Compute the center of rotation.
        pc = (self.fromState.x - self.r * self.fromState.s,
              self.fromState.y + self.r * self.fromState.c)

        # Check the arcs based on turning left/right and direction:
        if self.r > 0:
            # Turning left:  Check min (backleft) and max (frontright) radius
            if self.distance > 0:
                arc1 = (pc, self.fromState.box[1], self.toState.box[1])
                arc2 = (pc, self.fromState.box[3], self.toState.box[3])
            else:
                arc1 = (pc, self.toState.box[1], self.fromState.box[1])
                arc2 = (pc, self.toState.box[3], self.fromState.box[3])
        else:
            # Turning right: Check min (backright) and max (frontleft) radius
            if self.distance > 0:
                arc1 = (pc, self.toState.box[2], self.fromState.box[2])
                arc2 = (pc, self.toState.box[0], self.fromState.box[0])
            else:
                arc1 = (pc, self.fromState.box[2], self.toState.box[2])
                arc2 = (pc, self.fromState.box[0], self.toState.box[0])

        for wall in world_params.walls:
            if (SegmentCrossArc(wall, arc1) or
                SegmentCrossArc(wall, arc2)):
                return False
        return True


#
# LocalPlan (Abstract class). The base class for all local planners.
#
class LocalPlan(ABC):

    @abstractclassmethod
    def __init__(self, fromState: State, toState: State, car: CarParams):
        pass

    # Return the absolute length.
    @abstractclassmethod
    def Length(self) -> float:
        pass

    # Draw the local plan.
    @abstractclassmethod
    def Draw(self, fig: Visualization, color, **kwargs):
        pass

    @abstractclassmethod
    def DrawSimple(self, fig: Visualization, color, **kwargs):
        pass

    # Check whether all intermediate points and arcs are valid
    @abstractclassmethod
    def Valid(self, world: WorldParams) -> bool:
        pass

    @abstractclassmethod
    def IntermediateState(self, d: float) -> State:
        pass

    @abstractclassmethod
    def CriticalStates(self) -> List[State]:
        pass


#
#   2Arc Local Plan.  I'm using two arcs, but please feel free to create
#   whatever local planner you prefer.
#
class LocalPlan2Arc:
    def __init__(self, fromState: State, toState: State, car: CarParams):
        self.car = car

        # Compute the connection.
        (midState, arc1, arc2), r = self.Compute2ArcConnection(fromState, toState)
        self.valid_connection = r

        # Save the information.
        self.fromState = fromState
        self.midState  = midState
        self.toState   = toState
        self.arc1      = arc1
        self.arc2      = arc2

    def Compute2ArcConnection(self, fromState: State, toState: State) -> \
            Tuple[Tuple[Optional[State], Optional[Arc], Optional[Arc]], bool]:
        # Grab the starting and final coordinates.
        (x1, x2) = (fromState.x, toState.x)
        (y1, y2) = (fromState.y, toState.y)
        (t1, t2) = (fromState.t, toState.t)
        (s1, s2) = (fromState.s, toState.s)
        (c1, c2) = (fromState.c, toState.c)

        # COMPUTATION.  I find it useful to compute an turning radius
        # (or inverse thereof).  It is very useful to allow the local
        # planner to back up!!  Along one arc, or along all arcs...
        # You should know

        dx = x2 - x1
        dy = y2 - y1
        s = s1 + s2                     # Sum of Sins
        c = c1 + c2                     # Sum of Cos's
        u = s*s + c*c - 4               # a term of the quadratic
        v = c*dy - s*dx                 # -b term of the quadratic
    
        # Check for zero steering (infinite radius).
        if u == 0 and v == 0:
            # Straight line!
            invR = 0
            tm = t1                         # Theta at mid point
            xm = 0.5*(x1+x2)                # X coordinate at mid point
            ym = 0.5*(y1+y2)                # Y coordinate at mid point
            d1 = 0.5*np.sqrt(dx*dx + dy*dy) # Distance on first arc
            d2 = d1                         # Distance on second arc

        # Radius exists
        else:
            # Only 1 solution for radius
            if u == 0:
                invR = -(2 * (dx*s - dy*c)) / (dx*dx + dy*dy)
                
            # 2 solutions for radius
            else:
                det = np.sqrt(              # Determinant of the quadratic
                    -dy*dy*(s*s - 4) - dx*dx*(c*c - 4) - 2*c*s*dx*dy)

                # Pick the smallest radius that satisfies the max sterring angle
                # if no radius satisfies this, then this connection is invalid.
                invr1 = u/(v+det)
                invr2 = u/(v-det)
                abs1 = abs(invr1)
                abs2 = abs(invr2)
                invrmax = 1 / self.car.rmin
                if abs1 > invrmax:
                    if abs2 > invrmax:     # Neither is in range
                        return (None, None, None), False
                    invR = invr2            # Only invr2 is in range
                else:
                    if abs2 > invrmax:     # Only invr1 is in range
                        invR = invr1

                    # Both are in range. Use the smaller radius (so larger inv)
                    else:
                        invR = invr1 if abs1 > abs2 else invr2
            r = 1 / invR

            # Single arc
            if v == 0:
                tm = t2
                xm = x2
                ym = y2
                d1 = r * AngleDiff(t2, t1)
                d2 = 0

            # Double arc
            else:
                # Center of arc 1
                cx1 = x1 - r * s1
                cy1 = y1 + r * c1

                # Center of arc 2
                cx2 = x2 + r * s2
                cy2 = y2 - r * c2

                # Compute angle of tangent line at intersection. Start with slope
                # between the arc centers then phase by pi/2 depending on the
                # car direction (side of the radius).
                tm = np.arctan2(cy2 - cy1, cx2 - cx1) + np.sign(r) * np.pi/2

                # Compute arc intersection
                xm = 0.5 * (cx1 + cx2)
                ym = 0.5 * (cy1 + cy2)

                # Compute arc lengths from angle difference
                d1 = r * AngleDiff(tm, t1)
                d2 = -r * AngleDiff(t2, tm)

        # Return the mid state and two arcs.  Again, you may choose
        # differently, but the below is my approach.
        r = None if not invR else 1 / invR
        midState = State(xm, ym, tm, self.car)
        arc1     = Arc(fromState, midState, d1,  r)
        arc2     = Arc(midState , toState,  d2, None if not r else -r)
        return (midState, arc1, arc2), True

    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Arc1 %s\n Arc2 %s\n => Length %5.2f>" %
                (self.arc1, self.arc2, self.Length()))

    # Return the absolute length.
    def Length(self) -> float:
        d1 = 0 if not self.arc1 else self.arc1.Length()
        d2 = 0 if not self.arc2 else self.arc2.Length()
        return d1 + d2

    # Draw the local plan (showing the mid state and two arcs).
    def Draw(self, fig: Visualization, color, **kwargs):
        self.midState.Draw(fig, color, **kwargs)
        self.arc1.Draw(self.car, fig, color, **kwargs)
        self.arc2.Draw(self.car, fig, color, **kwargs)

    def DrawSimple(self, fig: Visualization, color, **kwargs):
        plt.plot((self.fromState.x, self.toState.x), 
                 (self.fromState.y, self.toState.y), color, **kwargs)

    ############################################################
    # PRM Functions:
    # Check whether the midpoint is in free space and
    # both arcs are valid (turning radius and collisions).
    def Valid(self, world: WorldParams) -> bool:
        return (self.valid_connection and
                self.midState.InFreeSpace(world) and
                self.arc1.Valid(world, self.car.rmin) and
                self.arc2.Valid(world, self.car.rmin))

    def IntermediateState(self, d: float) -> State:
        d1 = 0 if not self.arc1 else self.arc1.Length()
        d2 = 0 if not self.arc2 else self.arc2.Length()

        if d < d1:
            return self.arc1.IntermediateState(d * np.sign(self.arc1.distance), self.car)
        if d < d1 + d2:
            d -= d1
            return self.arc2.IntermediateState(d * np.sign(self.arc2.distance), self.car)
        return self.toState

    def CriticalStates(self) -> List[State]:
        if self.midState:
            return [self.midState]
        return []


#
# Local Plan 3-Arc
#
class LocalPlan3Arc(LocalPlan):
    def __init__(self, fromState: State, toState: State, car: CarParams):
        self.car = car

        # Compute the connection.
        (pointState, stopState, arc1, arc2, arc3), l = self.ComputeConnection(fromState, toState)
        self.length    = l

        # Save the information.
        self.fromState = fromState
        self.pointState = pointState
        self.stopState = stopState
        self.toState   = toState
        self.arc1      = arc1
        self.arc2      = arc2
        self.arc3      = arc3

    def ComputeConnection(self, fromState: State, toState: State) -> \
                            Tuple[Tuple[State, State, Arc, Arc, Arc], bool]:
        # Grab the starting and final coordinates.
        (x1, x2) = (fromState.x, toState.x)
        (y1, y2) = (fromState.y, toState.y)
        (t1, t2) = (fromState.t, toState.t)
        (s1, s2) = (fromState.s, toState.s)
        (c1, c2) = (fromState.c, toState.c)

        # rad_base = wheelbase / tansteermax  # Assuming the car is able to turn
        rad_base = self.car.rmin

        # Initialize values to minimize
        min_length = np.inf
        params = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)

        # Minimize over all 4 configurations (+/- r and forwards vs. backwards)
        for r in [rad_base, -rad_base]:
            # Center of circle 1
            cx1 = x1 - r * s1
            cy1 = y1 + r * c1

            # Center of circle 2
            cx2 = x2 - r * s2
            cy2 = y2 + r * c2

            dy = cy2 - cy1
            dx = cx2 - cx1
            
            ds = np.sqrt(dy**2 + dx**2)

            for phase in [0, np.pi]:
                tp = np.arctan2(dy, dx)
                tp = AngleDiff(tp + phase, 0)       # Adjust by phase

                dp = r * AngleDiff(tp, t1)          # Distance to point state
                dt = r * AngleDiff(t2, tp)
                l = abs(dp) + ds + abs(dt)
                if l < min_length:
                    min_length = l
                    params = (l, r, cx1, cy1, tp, dp, ds * np.cos(phase), dt)

        l, r, cx1, cy1, tp, dp, ds, dt = params

        # Arc 1 (fromState -> point)
        xp = cx1 + r * np.cos(tp - np.pi / 2)
        yp = cy1 + r * np.sin(tp - np.pi / 2)
        pointState = State(xp, yp, tp, self.car)
        arc1 = Arc(fromState, pointState, dp, r)

        # Arc 2 (point -> stop)
        (cp, sp) = (pointState.c, pointState.s)
        stopState = State(xp + ds * cp, yp + ds * sp, tp, self.car)
        arc2 = Arc(pointState, stopState, ds, None)

        # Arc 3 (stop -> toState)
        arc3 = Arc(stopState, toState, dt, r)

        return (pointState, stopState, arc1, arc2, arc3), l


    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self) -> str:
        return ("<Arc1 %s\n Arc2 %s\n Arc3 %s\n Arc4 %s\n => Length %5.2f>" %
                (self.arc1, self.arc2, self.arc3, self.Length()))

    # Return the absolute length.
    def Length(self) -> float:
        return self.length

    def __len__(self) -> float:
        return self.length

    # Draw the local plan (showing the mid state and two arcs).
    def Draw(self, fig: Visualization, color, **kwargs):
        self.pointState.Draw(fig, color, **kwargs)
        self.stopState.Draw(fig, color, **kwargs)
        self.arc1.Draw(self.car, fig, color, **kwargs)
        self.arc2.Draw(self.car, fig, color, **kwargs)
        self.arc3.Draw(self.car, fig, color, **kwargs)

    def DrawSimple(self, fig: Visualization, color, **kwargs):
        plt.plot((self.fromState.x, self.toState.x), 
                 (self.fromState.y, self.toState.y), color, **kwargs)

    ############################################################
    # PRM Functions:
    # Check whether the midpoint is in free space and
    # both arcs are valid (turning radius and collisions).
    def Valid(self, world: WorldParams) -> bool:
        car = self.car
        return (self.pointState.InFreeSpace(world) and 
                self.stopState.InFreeSpace(world) and
                self.arc1.Valid(world, car.rmin) and 
                self.arc2.Valid(world, car.rmin) and 
                self.arc3.Valid(world, car.rmin))

    def IntermediateState(self, d: float) -> State:
        d1 = self.arc1.Length()
        d2 = self.arc2.Length()
        d3 = self.arc3.Length()

        if d < d1:
            return self.arc1.IntermediateState(d * np.sign(self.arc1.distance), self.car)
        if d < d1 + d2:
            d -= d1
            return self.arc2.IntermediateState(d * np.sign(self.arc2.distance), self.car)
        if d < d1 + d2 + d3:
            d -= d1 + d2
            return self.arc3.IntermediateState(d * np.sign(self.arc3.distance), self.car)
        return self.toState

    def CriticalStates(self) -> List[State]:
        return [self.pointState, self.stopState]


MAX_DIST = 10
DISTANCE_FROM_GOAL = 9      # Must be > than ~8 to capture all cases

#
# Local Plan 4-Arc
#
class LocalPlan4Arc(LocalPlan2Arc):
    def __init__(self, fromState: State, toState: State, car: CarParams):
        self.car = car

        # Compute the connection.
        (pointState, stopState, midState, arc1, arc2, arc3, arc4), r = self.ComputeConnection(fromState, toState)
        self.valid_connection = r

        # Save the information.
        self.fromState = fromState
        self.pointState = pointState
        self.stopState = stopState
        self.midState  = midState
        self.toState   = toState
        self.arc1      = arc1
        self.arc2      = arc2
        self.arc3      = arc3
        self.arc4      = arc4

    def ComputeConnection(self, fromState: State, toState: State) -> \
            Tuple[Tuple[Optional[State], Optional[State], State, Optional[Arc], Optional[Arc], Arc, Arc], bool]:
        r = False       # Default to running 2 Arc if close enough

        # Too far away, turn towards the target and drive straight for a bit.
        if fromState.EuclideanDistance(toState) > MAX_DIST:
            # Grab the starting and final coordinates.
            (x1, x2) = (fromState.x, toState.x)
            (y1, y2) = (fromState.y, toState.y)

            # Determine which side to turn towards for straight-drive step.
            tp = np.arctan2(y2-y1, x2-x1)
            if tp < 0:
                tansteer = self.car.tansteermax
            else:
                tansteer = -self.car.tansteermax

            # If turning by tansteer is invalid, try turning by -tansteer
            pointState, stopState, arc1, arc2 = self.ComputeStraightDriveConnection(
                fromState, toState, tp, tansteer, DISTANCE_FROM_GOAL)

            # 2 Arc (stop -> toState)
            (midState, arc3, arc4), r = self.Compute2ArcConnection(stopState, toState)

        # Already close enough, just perform a 2 Arc
        else:
            # 2 Arc (fromState -> toState)
            (midState, arc3, arc4), r = self.Compute2ArcConnection(fromState, toState)
            pointState = None
            stopState = None
            arc1 = None
            arc2 = None

        return (pointState, stopState, midState, arc1, arc2, arc3, arc4), r

    def ComputeStraightDriveConnection(self, fromState: State, toState: State, 
                                        tp: float, r: float, goal_dist: float) \
                                        -> Tuple[State, State, Arc, Arc]:
        # Center of circle 1
        cx1 = fromState.x - r * fromState.s
        cy1 = fromState.y + r * fromState.c

        # Arc 1 (fromState -> point)
        dp = r * AngleDiff(tp, fromState.t)     # Distance to point state
        xp = cx1 + r * np.cos(tp - np.pi / 2)
        yp = cy1 + r * np.sin(tp - np.pi / 2)
        pointState = State(xp, yp, tp, self.car)
        arc1 = Arc(fromState, pointState, dp, r)

        # Arc 2 (point -> stop)
        ds = max(0, pointState.Distance(toState) - goal_dist)
        (cp, sp) = (pointState.c, pointState.s)
        stopState = State(xp + ds * cp, yp + ds * sp, tp, self.car)
        arc2 = Arc(pointState, stopState, ds, None)

        return pointState, stopState, arc1, arc2

    ############################################################
    # Utilities:
    # In case we want to print the state.
    def __repr__(self):
        return ("<Arc1 %s\n Arc2 %s\n Arc3 %s\n Arc4 %s\n => Length %5.2f>" %
                (self.arc1, self.arc2, self.arc3, self.arc4, self.Length()))

    # Return the absolute length.
    def Length(self) -> float:
        if not self.valid_connection:
            return np.inf
        return ((0 if not self.arc1 else self.arc1.Length()) + 
                (0 if not self.arc2 else self.arc2.Length()) + 
                self.arc3.Length() + 
                self.arc4.Length())

    # Draw the local plan (showing the mid state and two arcs).
    def Draw(self, fig: Visualization, color, **kwargs):
        if self.valid_connection:
            if self.pointState:
                self.pointState.Draw(fig, color, **kwargs)
            if self.stopState:
                self.stopState.Draw(fig, color, **kwargs)
            self.midState.Draw(fig, color, **kwargs)
            if self.arc1:
                self.arc1.Draw(self.car, fig, color, **kwargs)
            if self.arc2:
                self.arc2.Draw(self.car, fig, color, **kwargs)
            self.arc3.Draw(self.car, fig, color, **kwargs)
            self.arc4.Draw(self.car, fig, color, **kwargs)
        else:
            if self.pointState:
                self.pointState.Draw(fig, color, **kwargs)
            if self.stopState:
                self.stopState.Draw(fig, color, **kwargs)
            if self.arc1:
                self.arc1.Draw(self.car, fig, color, **kwargs)
            if self.arc2:
                self.arc2.Draw(self.car, fig, color, **kwargs)

    def DrawSimple(self, fig: Visualization, color, **kwargs):
        plt.plot((self.fromState.x, self.toState.x), 
                 (self.fromState.y, self.toState.y), color, **kwargs)

    ############################################################
    # PRM Functions:
    # Check whether the midpoint is in free space and
    # both arcs are valid (turning radius and collisions).
    def Valid(self, world: WorldParams) -> bool:
        rmin = self.car.rmin
        return (self.valid_connection and
                (not self.pointState or self.pointState.InFreeSpace(world)) and
                (not self.stopState or self.stopState.InFreeSpace(world)) and
                self.midState.InFreeSpace(world) and
                (not self.arc1 or self.arc1.Valid(world, rmin)) and
                (not self.arc2 or self.arc2.Valid(world, rmin)) and
                self.arc3.Valid(world, rmin) and
                self.arc4.Valid(world, rmin))

    def IntermediateState(self, d: float) -> State:
        d1 = 0 if not self.arc1 else self.arc1.Length()
        d2 = 0 if not self.arc2 else self.arc2.Length()
        d3 = self.arc3.Length()
        d4 = self.arc4.Length()

        if d < d1:
            return self.arc1.IntermediateState(d * np.sign(self.arc1.distance), self.car)
        if d < d1 + d2:
            d -= d1
            return self.arc2.IntermediateState(d * np.sign(self.arc2.distance), self.car)
        if d < d1 + d2 + d3:
            d -= d1 + d2
            return self.arc3.IntermediateState(d * np.sign(self.arc3.distance), self.car)
        if d < d1 + d2 + d3 + d4:
            d -= d1 + d2 + d3
            return self.arc4.IntermediateState(d * np.sign(self.arc4.distance), self.car)
        return self.toState

    def CriticalStates(self) -> List[State]:
        out = []
        if self.pointState:
            out.append(self.pointState)
        if self.stopState:
            out.append(self.stopState)
        out.append(self.midState)
        return out
