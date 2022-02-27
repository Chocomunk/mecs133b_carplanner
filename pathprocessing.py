from planners import Node
from carstate import State
from localplanners import LocalPlan
from visualization import visualization
from params import WorldParams, CarParams

#
#  Post Process the Path
#
class PathProcessor:

    def __init__(self, LocalPlanner: type[LocalPlan], world: WorldParams, car: CarParams):
        self.LocalPlanner = LocalPlanner
        self.world = world
        self.car = car

    def UniqueStates(self, path: list[Node]) -> list[State]:
        # Initialize the state list.
        states = [path[0].state]

        # Add all arc points (if they are unique).
        for i in range(1, len(path)):
            plan = self.LocalPlanner(path[i-1].state, path[i].state, self.car)
            for crit_state in plan.CriticalStates():
                if (states[-1].Distance(crit_state) > 0.01):
                    states.append(crit_state)

        # Return the full list.
        return states

    def TrimStates(self, states: list[State]) -> list[State]:
        # Remove unnecessary intermediate states
        final_states = [states[0]]
        prev_state = states[0]
        last_state = None
        for state in states[1:]:
            # Once you find a node that can't be reached from the previously-added
            # node, add the latest reachable node to the final_path, and start again
            # from there
            plan = self.LocalPlanner(prev_state, state, self.car)

            if not plan.Valid(self.world):
                final_states.append(last_state)
                prev_state = last_state
            last_state = state
        final_states.append(states[-1])
        return final_states

    def VerifyPath(self, path: list[Node]) -> list[bool]:
        for i in range(1, len(path)):
            plan = self.LocalPlanner(path[i-1].state, path[i].state, self.car)
            if not plan.Valid(self.world):
                return False
        return True
        
    def PostProcess(self, path: list[Node]) -> list[Node]:
        # Grab all states, including the intermediate states between arcs.
        # states = UniqueStates(path)
        states = [node.state for node in path]
        
        # Check whether we can skip states.
        states = self.TrimStates(states)

        # Rebuild and return the path (list of nodes).
        return [Node(state) for state in states]

    def DrawPath(self, path: list[Node], fig: visualization, color, check_steps=False, **kwargs):
        # Draw the individual local plans
        for i in range(len(path)-1):
            plan = self.LocalPlanner(path[i].state, path[i+1].state, self.car)
            plan.Draw(fig, color, **kwargs)
            if check_steps:
                path[i+1].state.Draw(fig, 'm', **kwargs)
                fig.ShowFigure()
                input("Valid={0}. (press enter to continue)".format(plan.Valid(self.world)))

        # Print the unique path elements.
        print("Unique steps in path:")
        for (i, state) in enumerate(self.UniqueStates(path)):
            print(i, state)
