import numpy as np
import random

# Pybeast imports
from pybeast.core.agents.neuralanimat import EvoFFNAnimat
from pybeast.core.evolve.population import Population, Group
from pybeast.core.evolve.geneticalgorithm import GeneticAlgorithm, GASelectionType
from pybeast.core.sensors.sensor import (
    SignalSensor,
    ProximitySensor,
)
from pybeast.core.simulation import Simulation
from pybeast.core.world.worldobject import WorldObject

IsDemo = True
GUIName = "EvoBat"
SimClassName = "EvoBatSimulation"


class Food(WorldObject):
    """Food source that bats can find."""

    def __init__(self):
        super().__init__()
        self.SetRadius(30.0)
        self.SetResetRandom(True)
        self.SetColour(0.0, 0.8, 0.2, 1.0)
        self.id = id(self)

    def getAward(self):
        """Get the award for the food object."""
        if self.myWorld is None:
            return 0

        # Count how many bats are nearby
        nearby_bats = 0
        for obj in self.myWorld.Get(EvoBat):
            if (
                isinstance(obj, EvoBat)
                and (obj.GetLocation() - self.GetLocation()).GetLength() <= 20
            ):
                nearby_bats += 1
        return nearby_bats

    def __del__(self):
        """Destructor."""
        pass


class EvoBat(EvoFFNAnimat):
    """Bat that can evolve to use signals to communicate food locations."""

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

        # Add sensors
        self.AddSensor(
            "foodR", ProximitySensor(Food, np.pi / 2, 80, np.pi / 8, simple=True)
        )
        self.AddSensor(
            "foodL", ProximitySensor(Food, np.pi / 2, 80, -np.pi / 8, simple=True)
        )
        self.AddSensor("signal", SignalSensor())
        self.SetInteractionRange(200.0)

        self.foodFound = False
        self.foodBonus = 0.0
        self.stamina = 5.0

        # Initialize neural network
        self.AddFFNBrain(hidden=4, inputs=3, outputs=3)

        # Set inputs to the neural network as (other_sensors) + (2*signals) for old method
        # self.AddFFNBrain(hidden=4, inputs=4, outputs=3)

    def Update(self):
        """Update bat state before control."""
        super().Update()

    def Control(self):
        """Process neural network outputs to control the bat."""
        outputs = self.GetBrainOutput()

        if not self.foodFound and self.stamina > 0:
            self.stamina -= 0.1

        # First two outputs control movement
        self.controls["left"] = outputs[0]
        self.controls["right"] = outputs[1]
        # Third output controls signal emission
        # if outputs[2] > 0 and self.foodFound:
        if outputs[2] > 0:
            if not self.IsTransmitting():
                self.SetSignalValue(1.0)
                self.StartTransmitting()
        else:
            if self.IsTransmitting():
                self.StopTransmitting()

    def Reset(self):
        """Reset bat state for new assessment."""
        self.foodFound = False
        self.foodBonus = 0.0
        self.stamina = 5.0
        if self.IsTransmitting():
            self.StopTransmitting()

        # Ensure minimum distance from food sources
        if self.myWorld is not None:
            min_distance = 200.0
            food_objects = self.myWorld.Get(Food)

            if food_objects:
                # Try repositioning up to 10 times
                for i in range(10):
                    too_close = False
                    for food in food_objects:
                        distance = (self.GetLocation() - food.GetLocation()).GetLength()
                        if distance < min_distance:
                            too_close = True
                            break

                    if too_close:
                        if self.myWorld:
                            self.SetLocation(self.myWorld.RandomLocation())
                            self.SetOrientation(np.random.uniform(0, 2 * np.pi))
                    else:
                        break

        super().Reset()

    def OnCollision(self, obj):
        """Handle collision with food."""
        if isinstance(obj, Food):
            self.foodFound = True
            self.foodBonus += 0.001
            factor = obj.getAward()
            if factor > 1:
                self.foodBonus += 0.1 * (2 ** (float(factor) / 2))

    def GetFitness(self) -> float:
        """Calculate fitness with group component and helper rewards."""
        # An award for finding food
        # individual_fitness = 3.0 if self.foodFound else 0.0

        # Get the food factor from the food object
        food_factor = self.foodBonus

        # total_fitness = food_factor + individual_fitness + self.stamina
        total_fitness = food_factor + self.stamina

        # Ensure fitness doesn't go negative
        return max(0.0, total_fitness)


class EvoBatSimulation(Simulation):
    """Simulation for evolving bats that communicate with signals."""

    def __init__(self):
        super().__init__("EvoBat Simulation")
        random.seed(42)
        np.random.seed(42)

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
            0.25, 0.35, selection=GASelectionType.GA_TOURNAMENT
        )

        self.Add("bats", Population(pop_size, EvoBat, self.theGA))
        self.Add("food", Group(2, Food))

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
    simulation = EvoBatSimulation()
    simulation.RunSimulation(render=True)
