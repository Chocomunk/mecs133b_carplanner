import numpy as np


######################################################################
#
#   Car Definitions
#
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
    invrmax = tansteermax / wheelbase


######################################################################
#
#   World Definitions
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


class SmallWorld(WorldParams):
    pass


class ZigZagWorld(WorldParams):
    (wroad)                  = 20                            # Road
    (xspace, lspace, wspace) = (5, 6, 2.5)                  # Parking Space
    (xmin, ymin, xmax, ymax) = (0, 0, 40, wroad+wspace)     # Overall boundary

    # Construct the walls.
    walls = (((xmin         , ymin        ), (xmax         , ymin        )),
            ((xmax         , ymin        ), (xmax         , wroad       )),
            ((xmax         , wroad       ), (xspace+lspace, wroad       )),
            ((xspace+lspace, wroad       ), (xspace+lspace, wroad+wspace)),
            ((xspace+lspace, wroad+wspace), (xspace       , wroad+wspace)),
            ((xspace       , wroad+wspace), (xspace       , wroad       )),
            ((xspace       , wroad       ), (xmin         , wroad       )),
            ((xmin         , wroad       ), (xmin         , 0           )),
            ((xmin         , wroad/4     ), (xmax * .75   , wroad/4     )),
            ((xmin + (xmax - xmin) * .25, wroad/2), (xmax, wroad/2      )),
            ((xmin         , 3*wroad/4   ), (xmax * .75   , 3*wroad/4   ))
            )


class WallWorld(WorldParams):
    (wroad)                  = 20                            # Road
    (xspace, lspace, wspace) = (5, 6, 2.5)                  # Parking Space
    (xmin, ymin, xmax, ymax) = (0, 0, 40, wroad+wspace)     # Overall boundary

    # Construct the walls.
    walls = (((xmin         , ymin        ), (xmax         , ymin        )),
            ((xmax         , ymin        ), (xmax         , wroad       )),
            ((xmax         , wroad       ), (xspace+lspace, wroad       )),
            ((xspace+lspace, wroad       ), (xspace+lspace, wroad+wspace)),
            ((xspace+lspace, wroad+wspace), (xspace       , wroad+wspace)),
            ((xspace       , wroad+wspace), (xspace       , wroad       )),
            ((xspace       , wroad       ), (xmin         , wroad       )),
            ((xmin         , wroad       ), (xmin         , 0           )),
            ((xmin + 5     , ymin + 5    ), (xmax - 5     , ymin + wroad - 5))
            )


class BlockWorld(WorldParams):
    (wroad)                  = 20                            # Road
    (xspace, lspace, wspace) = (5, 6, 2.5)                  # Parking Space
    (xmin, ymin, xmax, ymax) = (0, 0, 40, wroad+wspace)     # Overall boundary

    # Construct the walls.
    walls = (((xmin         , ymin        ), (xmax         , ymin        )),
            ((xmax         , ymin        ), (xmax         , wroad       )),
            ((xmax         , wroad       ), (xspace+lspace, wroad       )),
            ((xspace+lspace, wroad       ), (xspace+lspace, wroad+wspace)),
            ((xspace+lspace, wroad+wspace), (xspace       , wroad+wspace)),
            ((xspace       , wroad+wspace), (xspace       , wroad       )),
            ((xspace       , wroad       ), (xmin         , wroad       )),
            ((xmin         , wroad       ), (xmin         , 0           )),
            ((xmin + 5     , ymin + 5    ), (xmin + 5     , ymin + wroad - 5)),
            ((xmin + 5     , ymin + 5    ), (xmax - 7     , ymin + 5    )),
            ((xmax - 7     , ymin + wroad - 5), (xmin + 5 , ymin + wroad - 5)),
            ((xmax - 7     , ymin + wroad - 5), (xmax - 7 , ymin + 5    ))
            )
