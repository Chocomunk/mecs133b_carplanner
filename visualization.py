import matplotlib.pyplot as plt


from params import WorldParams


class DrawParams():
    # Spacing for drawing/testing
    ddraw = 0.5


######################################################################
#
#   Visualization
#
class Visualization:
    def __init__(self):
        # Clear and show.
        self.ClearFigure()
        self.ShowFigure()

    def ClearFigure(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and prepare the axes.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim( -9, 15)
        plt.gca().set_ylim(-12, 12)
        plt.gca().set_aspect('equal')

    def ShowParkingSpot(self, world_params: WorldParams):
        # Define the region (axis limits).
        plt.gca().set_xlim(world_params.xmin, world_params.xmax)
        plt.gca().set_ylim(world_params.ymin, world_params.ymax)

        # Show the walls.
        for wall in world_params.walls:
            plt.plot([wall[0][0], wall[1][0]],
                    [wall[0][1], wall[1][1]], 'k', linewidth=2)

        # Mark the locations.
        plt.gca().set_xticks(list(set([wall[0][0] for wall in world_params.walls])))
        plt.gca().set_yticks(list(set([wall[0][1] for wall in world_params.walls])))

    def ShowFigure(self):
        # Show the plot.
        plt.pause(0.001)
