import numpy as np
import random

# Pybeast imports
from pybeast.core.agents.neuralanimat import EvoFFNAnimat, EvoDNNAnimat
from pybeast.core.evolve.population import Population, Group
from pybeast.core.evolve.geneticalgorithm import GeneticAlgorithm, GASelectionType
from pybeast.core.sensors.sensor import (
    SignalSensor,
    ProximitySensor,
    NearestAngleSensor,
)
from pybeast.core.simulation import Simulation
from pybeast.core.world.worldobject import WorldObject
from pybeast.core.utils.colours import ColourPalette, ColourType
from pybeast.core.utils.vector2D import Vector2D

IsDemo = True
GUIName = "EvoBat2"
SimClassName = "EvoBatSimulation2"


class Food(WorldObject):
    """Food source that bats can find."""

    def __init__(self):
        super().__init__()
        self.SetRadius(20.0)
        self.SetResetRandom(True)
        self.SetColour(0.0, 0.8, 0.2, 1.0)
        self.id = id(self)

    def __del__(self):
        """Destructor."""
        pass


class EvoBat(EvoFFNAnimat):
    # class EvoBat(EvoDNNAnimat):
    """Bat that can evolve to use signals to communicate food locations."""

    def __init__(self):
        super().__init__()

        # Basic animat settings
        self.foodFound = False
        self.SetSolid(False)
        self.SetRadius(10.0)
        self.SetMinSpeed(40.0)
        self.SetMaxSpeed(80.0)
        self.SetSignalAgent(True)
        self.SetMaxRotationSpeed(np.pi)

        # Configure signal properties
        self.SetSignalStrength(200.0)

        # Add sensors
        self.AddSensor(
            "foodR", ProximitySensor(Food, np.pi / 2, 80, np.pi / 8, simple=True)
        )
        self.AddSensor(
            "foodL", ProximitySensor(Food, np.pi / 2, 80, -np.pi / 8, simple=True)
        )
        self.AddSensor("signal", SignalSensor())
        self.SetInteractionRange(200.0)

        # Cooperation tracking
        self.signaled_after_finding = 0

        # Add dishonest signaling tracking
        self.dishonest_signals = 0

        # Initialize neural network
        self.AddFFNBrain(hidden=4, inputs=3, outputs=3)

        # Set inputs to the neural network as (other_sensors) + (2*signals) for old method
        #self.AddFFNBrain(hidden=4, inputs=4, outputs=3)

        # self.InitDNN(total=4, inputs=4, outputs=3)

    def Update(self):
        """Update bat state before control."""
        super().Update()

    def Control(self):
        """Process neural network outputs to control the bat."""
        outputs = self.GetBrainOutput()

        # First two outputs control movement
        self.controls["left"] = outputs[0]
        self.controls["right"] = outputs[1]
        # Third output controls signal emission
        if outputs[2] > 0:
            # if self.controls["signal"] > 0.5:

            # This is a bit hacky, but it works for now
            sensors = self.GetSensors()
            food_sensor_r = sensors["foodR"].GetOutput()
            food_sensor_l = sensors["foodL"].GetOutput()
            near_food = food_sensor_r > 0.5 or food_sensor_l > 0.5
            if not self.IsTransmitting():
                self.SetSignalValue(1.0)
                self.StartTransmitting()

            if self.IsTransmitting():
                # If signaling when near food, count as cooperation
                if near_food:
                    self.signaled_after_finding += 1
                # If signaling without being near food, count as dishonest
                elif not near_food:
                    self.dishonest_signals += 1
        else:
            if self.IsTransmitting():
                self.StopTransmitting()

    def Reset(self):
        """Reset bat state for new assessment."""
        self.foodFound = False
        self.signaled_after_finding = 0
        self.dishonest_signals = 0
        if self.IsTransmitting():
            self.StopTransmitting()
        super().Reset()

    def OnCollision(self, obj):
        """Handle collision with food."""
        if isinstance(obj, Food):
            self.foodFound = True

    def GetFitness(self) -> float:
        """Calculate fitness with group component and helper rewards."""
        # Base fitness from finding food
        individual_fitness = 5.0 if self.foodFound else 0.0

        # Reward for signaling after finding food (announcing discoveries)
        cooperation_bonus = self.signaled_after_finding * 2.0

        # Penalty for dishonest signaling (transmitting when not near food)
        dishonest_penalty = self.dishonest_signals * 1.0

        total_fitness = cooperation_bonus - dishonest_penalty + individual_fitness

        # Ensure fitness doesn't go negative
        return max(0.0, total_fitness)


class EvoBatSimulation2(Simulation):
    random.seed(42)
    np.random.seed(42)
    """Simulation for evolving bats that communicate with signals."""

    def __init__(self):
        super().__init__("EvoBat Simulation 2")

        # create/clear log.csv file
        with open("other/log.csv", "w") as f:
            f.write("Generation,avg_fitness,max_fitness\n")

        # Simulation parameters
        self.SetGenerations(-1)
        self.SetAssessments(1)
        self.SetTimeSteps(800)

        # Set up genetic algorithm
        pop_size = 30
        self.theGA = GeneticAlgorithm(
            0.25, 0.2, selection=GASelectionType.GA_TOURNAMENT
        )

        self.Add("bats", Population(pop_size, EvoBat, self.theGA))
        self.Add("food", Group(5, Food))

        # Logging settings
        self.sleepBetweenLogs = 0.0
        for k in ["Simulation", "Run", "Generation"]:
            self.whatToLog[k] = True
        self.whatToSave["Simulation"] = self.whatToSave["Run"] = self.whatToSave[
            "Generation"
        ] = True

    def LogEndGeneration(self):
        super().LogEndGeneration()

        avg_fitness = np.mean(self.contents["bats"].AverageFitnessScoreOfMembers())
        max_fitness = np.max(self.contents["bats"].AverageFitnessScoreOfMembers())

        self.logger.info(
            f"Average fitness: {avg_fitness:.2f}, Max fitness: {max_fitness:.2f}, "
        )

        # save the generation, max and average fitness to a log.csv file
        with open("other/log.csv", "a") as f:
            f.write(f"{self.generation+1},{avg_fitness},{max_fitness}\n")

    def CreateDataStructSimulation(self):
        self.data = {}

    def CreateDataStructRun(self):
        self.averageFitness = []
        self.maxFitness = []

    def SaveGeneration(self):
        avg_fitness = np.mean(self.contents["bats"].AverageFitnessScoreOfMembers())
        max_fitness = np.max(self.contents["bats"].AverageFitnessScoreOfMembers())

        self.averageFitness.append(avg_fitness)
        self.maxFitness.append(max_fitness)

    def SaveRun(self):
        self.data[f"Run{self.Run}_avg"] = self.averageFitness
        self.data[f"Run{self.Run}_max"] = self.maxFitness


if __name__ == "__main__":
    simulation = EvoBatSimulation2()
    simulation.RunSimulation(render=True)
