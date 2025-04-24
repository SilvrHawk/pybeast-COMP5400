# Third-party
import numpy as np

from pybeast.core.simulation import Simulation
from pybeast.core.world.drawable import Drawable

# Pybeast
from pybeast.core.evolve.population import Population, Group
from pybeast.core.world.worldobject import WorldObject
from pybeast.core.agents.neuralanimat import EvoFFNAnimat
from pybeast.core.agents.animat import Animat
from pybeast.core.sensors.sensor import ProximitySensor
from pybeast.core.utils.vector2D import Vector2D
from pybeast.core.utils.colours import ColourPalette, ColourType
from pybeast.core.evolve.geneticalgorithm import GeneticAlgorithm, Evolver, GASelectionType, MutationOperator


IsDemo = True
GUIName = "BatEvo"
SimClassName = "BatBaseSimulation"


class TargetObject(WorldObject):
    """
    A central target object that agents will interact with.
    """

    def __init__(self):
        super().__init__()  # Larger radius to make it easier to hit
        self.SetResetRandom(True)
        self.SetRadius(30.0)
        self.SetColour(0.9, 0.1, 0.1, 1.0)  # Bright red

    def __del__(self):
        pass


class SignalAgent(EvoFFNAnimat):
    """
    An agent that emits signals when touching the target.
    """

    def __init__(self):
        super().__init__()

        # Set agent properties
        self.SetMinSpeed(50.0)
        self.SetMaxSpeed(80.0)
        self.SetTimeStep(0.05)
        self.SetRadius(15.0)
        self.SetMaxRotationSpeed(np.pi)

        # Configure signal properties
        self.SetSignalAgent(True)
        self.SetSignalStrength(100.0)
        self.AddFFNBrain(4)

        # Track whether we're touching the target and times touched
        self.touching_target = False
        self.times_transmitted = 0

        # Set a cyan color for the agent
        Drawable.SetColour(self, 0.0, 0.7, 1.0, 1.0)

    def Control(self):
        """
        Control method that handles random movement and signal emission.
        """
        super().Control()

        # Movement from EvoFFNAnimat with bias
        for n, k in enumerate(self.controls.keys()):
            self.controls[k] = self.controls[k] + 0.5

        # Reset touching_target flag at the start of each frame
        if self.touching_target == True:
            self.touching_target = False
        else:
            # If we were touching but no longer are, stop transmitting
            if self.IsTransmitting():
                self.StopTransmitting()
    
    def Reset(self):
        self.times_transmitted = 0
        super().Reset()

    def OnCollision(self, other):
        # Check if the object we collided with is our target
        if isinstance(other, TargetObject):
            # Mark that we're touching the target this frame
            self.touching_target = True

            # Start transmitting if not already
            if not self.IsTransmitting():
                
                # Increment the transmission count for fitness
                self.times_transmitted += 1
                self.SetSignalValue(1.0)
                self.StartTransmitting()

    def GetFitness(self) -> float:

        # Temporary fitness to judge how many times bats have transmitted
        return self.times_transmitted + 1

        #return 1/(self.times_transmitted+1)


class BatBaseSimulation(Simulation):
    """
    Simulation with a central target and a signal-emitting agent.
    """

    def __init__(self):
        super().__init__("Bat Base Simulation")

        self.SetGenerations(-1)
        self.SetAssessments(1)
        self.SetTimeSteps(500)

        self.theGA = GeneticAlgorithm( 0.25,0.1, selection = GASelectionType.GA_ROULETTE)
        self.theGA.SetSelection(GASelectionType.GA_ROULETTE)
        
        theBats =  Population(30, SignalAgent, self.theGA)
        theTarget = Group(5, TargetObject)

        self.Add("theBats", theBats)
        self.Add("theTarget", theTarget)

    def LogEndGeneration(self):

        super().LogEndGeneration()
        self.logger.info(f'Average fitness {self.avgFitness:.5f}')

    def CreateDataStructSimulation(self):
        self.data = {}

    def CreateDataStructRun(self):
        self.averageFitness = []

    def SaveGeneration(self):
        self.avgFitness = np.mean(self.contents['theBats'].AverageFitnessScoreOfMembers())
        self.avgFitness.append(self.avgFitness)

        return

# For standalone testing
if __name__ == "__main__":
    
    simulation = BatBaseSimulation()
