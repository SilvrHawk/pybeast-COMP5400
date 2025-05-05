# Third-party
import numpy as np
import random

from pybeast.core.simulation import Simulation
from pybeast.core.world.drawable import Drawable

# Pybeast
from pybeast.core.world.worldobject import WorldObject
from pybeast.core.agents.animat import Animat
from pybeast.core.sensors.sensor import SignalSensor
from pybeast.core.utils.vector2D import Vector2D

IsDemo = True
GUIName = "BatBase"
SimClassName = "BatBaseSimulation"


class Food(WorldObject):
    """
    A central target object that agents will interact with.
    """

    def __init__(self, l=Vector2D()):
        super().__init__(l)
        self.SetRadius(40.0)
        self.SetResetRandom(True)
        self.SetColour(0.0, 0.8, 0.2, 1.0)
        self.id = id(self)

    def __del__(self):
        pass


class SignalAgent(Animat):
    """
    An agent that emits signals when touching the target.
    """

    def __init__(self):
        super().__init__()

        # Basic animat settings
        self.SetSolid(False)
        self.SetRadius(10.0)
        self.SetMinSpeed(40.0)
        self.SetMaxSpeed(80.0)
        self.SetSignalAgent(True)
        self.SetMaxRotationSpeed(np.pi)

        # Configure signal properties
        self.SetSignalStrength(200.0)

        self.AddSensor("signal", SignalSensor())
        self.SetInteractionRange(200.0)

        self.touching_target = False

    def Control(self):
        """
        Control method that handles random movement and signal emission.
        """
        signal_value = 0.0
        angle_to_sender = None

        for k, v in self.received_signals.items():
            signal_value = v["value"]
            if "angle" in v:
                angle_to_sender = v["angle"]

        if self.touching_target == True:
            self.touching_target = False
        else:
            if self.IsTransmitting():
                self.StopTransmitting()

        if signal_value > 0 and angle_to_sender is not None:
            if angle_to_sender > 0:
                self.controls["right"] = 0.5 - signal_value * 0.5
                self.controls["left"] = 0.5 + signal_value * 0.3
            else:
                self.controls["right"] = 0.5 + signal_value * 0.3
                self.controls["left"] = 0.5 - signal_value * 0.5
        else:
            self.controls["left"] = 0.5
            self.controls["right"] = 0.5

    def OnCollision(self, other):

        if isinstance(other, Food):
            self.touching_target = True

            if not self.IsTransmitting():
                self.SetSignalValue(1.0)
                self.StartTransmitting()


class BatBaseSimulation(Simulation):
    """
    Simulation with multiple food objects and signal-emitting agents.
    """

    def __init__(self):
        super().__init__("Bat Base Simulation")
        self.SetTimeSteps(-1)
        np.random.seed(42)
        random.seed(42)

    def BeginAssessment(self):
        world_width = self.theWorld.GetWidth()
        world_height = self.theWorld.GetHeight()

        for _ in range(3):
            # Random position within the world boundaries with some padding
            x = np.random.uniform(40 * 2, world_width - 40 * 2)
            y = np.random.uniform(40 * 2, world_height - 40 * 2)
            food = Food(Vector2D(x, y))
            self.theWorld.Add(food)

        # Add signal agents at random positions
        for i in range(20):
            agent = SignalAgent()

            # Random position within the world boundaries
            x = np.random.uniform(10 * 2, world_width - 10 * 2)
            y = np.random.uniform(10 * 2, world_height - 10 * 2)
            agent.SetStartLocation(Vector2D(x, y))

            # Random orientation (0 to 2pi)
            orientation = np.random.uniform(0, 2 * np.pi)
            agent.SetStartOrientation(orientation)

            # Add the agent to the world
            self.theWorld.Add(agent)

        super().BeginAssessment()


if __name__ == "__main__":
    simulation = BatBaseSimulation()
