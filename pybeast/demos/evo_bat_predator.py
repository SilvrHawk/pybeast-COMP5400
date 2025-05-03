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
import random

IsDemo = True
GUIName = "BatPredatorEvo"
SimClassName = "BatBaseSimulation"

class Predator(EvoFFNAnimat):
    """
    A predator agent that can detect and chase the target.
    """

    def __init__(self, vocabSize = 3):
        super().__init__()

        # Set agent properties
        self.vocabSize = vocabSize
        self.SetMinSpeed(50.0)
        self.SetMaxSpeed(65.0)
        self.SetTimeStep(0.05)
        self.SetRadius(30.0)
        self.SetMaxRotationSpeed(np.pi)
        self.preyCaptured = 0

        # Configure signal properties
        self.SetSignalAgent(True)
        self.SetSignalStrength(100.0)
        self.AddFFNBrain(4)
        self.touching_target = False

        # Set a red color for the predator
        Drawable.SetColour(self, 1.0, 0.0, 0.0, 1.0)

    def Reset(self):
        self.preyCaptured = 0
        super().Reset()
    
    def Control(self):
        """
        Control method that handles random movement and signal emission.
        """
        super().Control()
        #print(self.GetReceivedSignals())
        value = any(signal.get('value') == 1.0 for signal in self.GetReceivedSignals().values())

        if value:
            #print("yes")
            self.SetSignalValue(2.0)
            self.StartTransmitting()
        else:
            #print("stop")
            self.SetSignalValue(0.0)
            self.StopTransmitting()

        # Movement from EvoFFNAnimat with bias
        for n, k in enumerate(self.controls.keys()):
            self.controls[k] = self.controls[k] + 0.5

    def OnCollision(self, other):
        # Check if the object we collided with is our target
        # if isinstance(other, TargetObject):
        #     # Mark that we're touching the target this frame
        #     self.touching_target = True

        #     # Start transmitting if not already
        #     if not self.IsTransmitting():
                
        #         # Increment the transmission count for fitness
        #         self.SetSignalValue(2.0)
        #         self.StartTransmitting()
        
        if isinstance(other, SignalAgent):
            # Increment the prey captured count
            self.preyCaptured += 1

    def GetFitness(self) -> float:

        return self.preyCaptured

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

    def __init__(self, vocabSize = 3):
        super().__init__()

        self.vocabSize = vocabSize
        # Set agent properties
        self.SetMinSpeed(50.0)
        self.SetMaxSpeed(80.0)
        self.SetTimeStep(0.05)
        self.SetRadius(15.0)
        self.SetMaxRotationSpeed(np.pi)

        # Configure signal properties
        self.SetSignalAgent(True)
        self.SetSignalStrength(200.0)
        self.AddFFNBrain(4)

        # Track whether we're touching the target and times touched
        self.touching_target = False
        self.times_transmitted = 0
        self.times_caught = 0

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

        value = any(signal.get('value') == 2.0 for signal in self.GetReceivedSignals().values())

        # Reset touching_target flag at the start of each frame
        if self.touching_target == True:
            self.touching_target = False
        else:
            # If we were touching but no longer are, stop transmitting
            if self.IsTransmitting():
                self.StopTransmitting()

        if value:
            #print("yes")
            self.SetSignalValue(3.0)
            self.StartTransmitting()
        else:
            #print("stop")
            self.SetSignalValue(0.0)
            self.StopTransmitting()
    
    def Reset(self):
        self.times_transmitted = 0
        self.times_caught = 0
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
        
        if isinstance(other, Predator):
            self.times_caught += 1
            self.location = self.myWorld.RandomLocation()

    def GetFitness(self) -> float:

        # Temporary fitness to judge how many times bats have transmitted
        return self.times_transmitted/self.times_caught if self.times_caught > 0 else 0.0


class BatBaseSimulation(Simulation):
    """
    Simulation with a central target and a signal-emitting agent.
    """
    random.seed(42)
    np.random.seed(42)

    def __init__(self):
        super().__init__("Bat Base Simulation")

        self.SetGenerations(-1)
        self.SetAssessments(1)
        self.SetTimeSteps(500)

        self.theGA = GeneticAlgorithm( 0.25,0.1, selection = GASelectionType.GA_ROULETTE)
        self.theGA.SetSelection(GASelectionType.GA_ROULETTE)

        self.gaPred = GeneticAlgorithm(0.25, 0.1, selection = GASelectionType.GA_ROULETTE)

        thePreds = Population(4, Predator, self.gaPred)        
        theBats =  Population(30, SignalAgent, self.theGA)
        theTarget = Group(5, TargetObject)

        self.Add("theBats", theBats)
        self.Add("theTarget", theTarget)
        self.Add("thePreds", thePreds)

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
