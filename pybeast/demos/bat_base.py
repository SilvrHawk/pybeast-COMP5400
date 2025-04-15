# Third-party
import numpy as np

from pybeast.core.simulation import Simulation
from pybeast.core.world.drawable import Drawable

# Pybeast
from pybeast.core.world.worldobject import WorldObject
from pybeast.core.agents.animat import Animat
from pybeast.core.sensors.sensor import ProximitySensor
from pybeast.core.utils.vector2D import Vector2D
from pybeast.core.utils.colours import ColourPalette, ColourType

IsDemo = True
GUIName = "BatBase"
SimClassName = "BatBaseSimulation"


class TargetObject(WorldObject):
    """
    A central target object that agents will interact with.
    """

    def __init__(self, l=Vector2D()):
        super().__init__(l, 0.0, 30.0)  # Larger radius to make it easier to hit
        self.SetColour(0.9, 0.1, 0.1, 1.0)  # Bright red

    def __del__(self):
        pass


class SignalAgent(Animat):
    """
    An agent that emits signals when touching the target.
    """

    def __init__(self):
        super().__init__(randomColour=False)

        self.SetMinSpeed(50.0)
        self.SetMaxSpeed(80.0)
        self.SetTimeStep(0.05)
        self.SetRadius(15.0)
        self.SetMaxRotationSpeed(np.pi)

        # Configure signal properties
        self.SetSignalStrength(100.0)

        # Track whether we're touching the target
        self.touching_target = False

        # Set a cyan color for the agent
        Drawable.SetColour(self, 0.0, 0.7, 1.0, 1.0)

    def Control(self):
        """
        Control method that handles random movement and signal emission.
        """
        signal_value = 0.0
        angle_to_sender = None

        # Process received signals
        for k, v in self.received_signals.items():
            signal_value = v["value"]
            if "angle" in v:
                angle_to_sender = v["angle"]

        # Reset touching_target flag at the start of each frame
        if self.touching_target == True:
            self.touching_target = False
        else:
            # If we were touching but no longer are, stop transmitting
            if self.IsTransmitting():
                self.StopTransmitting()

        # Turn away from the sender based on angle
        if signal_value > 0 and angle_to_sender is not None:
            if angle_to_sender > 0:
                self.controls["right"] = 0.5 + signal_value * 0.5
                self.controls["left"] = 0.5 - signal_value * 0.3
            else:
                self.controls["right"] = 0.5 - signal_value * 0.3
                self.controls["left"] = 0.5 + signal_value * 0.5
        else:
            # Default movement if no signal - just go straight
            self.controls["left"] = 0.5
            self.controls["right"] = 0.5

    def OnCollision(self, other):
        # Check if the object we collided with is our target
        if isinstance(other, TargetObject):
            # Mark that we're touching the target this frame
            self.touching_target = True

            # Start transmitting if not already
            if not self.IsTransmitting():
                self.SetSignalValue(1.0)
                self.StartTransmitting()


class BatBaseSimulation(Simulation):
    """
    Simulation with a central target and a signal-emitting agent.
    """

    def __init__(self):
        super().__init__("Bat Base Simulation")
        self.SetTimeSteps(-1)

    def BeginAssessment(self):
        # Add a target in the center of the world
        world_width = self.theWorld.GetWidth()
        world_height = self.theWorld.GetHeight()
        self.theWorld.Add(TargetObject(Vector2D(world_width / 2, world_height / 2)))

        # Add the signal agent at a random position
        agent = SignalAgent()
        agent2 = SignalAgent()
        agent3 = SignalAgent()

        start_x = world_width / 2 - 150.0
        start_y = world_height / 2
        agent.SetStartLocation(Vector2D(start_x, start_y))
        agent.SetStartOrientation(0.0)

        start_x = world_width / 2
        start_y = world_height / 2 - 100.0
        agent2.SetStartLocation(Vector2D(start_x, start_y))
        agent2.SetStartOrientation(np.pi / 2)

        start_x = world_width / 2 + 100.0
        start_y = world_height / 2 + 100.0
        agent3.SetStartLocation(Vector2D(start_x, start_y))
        agent3.SetStartOrientation(np.pi)

        self.theWorld.Add(agent)
        self.theWorld.Add(agent2)
        self.theWorld.Add(agent3)

        super().BeginAssessment()


# For standalone testing
if __name__ == "__main__":
    simulation = BatBaseSimulation()
