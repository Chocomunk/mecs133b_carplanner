import numpy as np

######################################################################
#
#   General/World Definitions
#
#   List of objects, start, goal, and parameters.
#

class WorldParams():
    (wroad)                  = 6                            # Road
    (xspace, lspace, wspace) = (5, 6, 2.5)                  # Parking Space
    (xmin, ymin, xmax, ymax) = (0, 0, 20, wroad+wspace)     # Overall boundary

    # Construct the walls.
    walls = (((xmin         , ymin        ), (xmax         , ymin        )),
            ((xmax         , ymin        ), (xmax         , wroad       )),
            ((xmax         , wroad       ), (xspace+lspace, wroad       )),
            ((xspace+lspace, wroad       ), (xspace+lspace, wroad+wspace)),
            ((xspace+lspace, wroad+wspace), (xspace       , wroad+wspace)),
            ((xspace       , wroad+wspace), (xspace       , wroad       )),
            ((xspace       , wroad       ), (xmin         , wroad       )),
            ((xmin         , wroad       ), (xmin         , 0           )))

# Define the car.  Outline and center of rotation.
class CarParams():
    (lcar, wcar) = (4, 2)           # Length/width
    lb           = 0.5              # Center of rotation to back bumper
    lf           = lcar - lb        # Center of rotation to front bumper
    wc           = wcar/2           # Center of rotation to left/right
    wheelbase    = 3                # Center of rotation to front wheels

    # Max steering angle.
    steermax    = np.pi/4
    tansteermax = np.tan(steermax)
    rmin = wheelbase / tansteermax