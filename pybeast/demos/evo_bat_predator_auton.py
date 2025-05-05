# Third-party
import numpy as np

from pybeast.core.simulation import Simulation
from pybeast.core.world.drawable import Drawable

# Pybeast
from pybeast.core.sensors.sensor import SignalSensor
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
GUIName = "BatPredatorEvoAuton"
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
        self.SetMaxSpeed(72.0)
        self.SetTimeStep(0.05)
        self.SetRadius(30.0)
        self.SetMaxRotationSpeed(np.pi)
        self.preyCaptured = 0
        self.times_transmitted = 0
        self.times_transmitted_false = 0

        # Configure signal properties
        self.SetSignalAgent(True)
        self.SetSignalStrength(150.0)
        self.touching_target = False
        self.transmitted_correctly = False
        self.transmitted_poorly = False

        self.AddSensor("signal", SignalSensor())
        self.SetInteractionRange(200.0)

        self.AddSensor(
            "preyR", ProximitySensor(EvoBat, np.pi / 2, 80, np.pi / 8, simple=True)
        )
        self.AddSensor(
            "preyL", ProximitySensor(EvoBat, np.pi / 2, 80, -np.pi / 8, simple=True)
        )

        self.AddFFNBrain(hidden=10, inputs=2+vocabSize, outputs=2+vocabSize)

        # Set a red color for the predator
        Drawable.SetColour(self, 1.0, 0.0, 0.0, 1.0)

    def Reset(self):
        print(self.times_transmitted_false)
        self.preyCaptured = 0
        self.times_transmitted = 0
        self.times_transmitted_false = 0
        self.transmitted_correctly = False
        self.transmitted_poorly = False
        super().Reset()
    
    def Control(self):
        """
        Control method that handles random movement and signal emission.
        """
        #super().Control()
        outputs = self.GetBrainOutput()

        # Movement from EvoFFNAnimat with bias
        # for n, k in enumerate(self.controls.keys()):
        #     self.controls[k] = 0.5 * (self.controls[k] + 1.0)

        self.controls["left"] = outputs[0]
        self.controls["right"] = outputs[1]
        signals = outputs[2:2+self.vocabSize]

        exp_logits = np.exp(signals - np.max(signals))
        prob = exp_logits / np.sum(exp_logits)
        signal_id = np.random.choice(self.vocabSize, p=prob)+1

        #print(self.GetReceivedSignals())
        #value = any(signal.get('value') == 1.0 for signal in self.GetReceivedSignals().values())

        if (sum(signals)/len(signals)) > 0.0:
            sensors = self.GetSensors()
            prey_sensor_r = sensors["preyR"].GetOutput()
            prey_sensor_l = sensors["preyL"].GetOutput()
            near_prey = prey_sensor_r > 0.25 or prey_sensor_l > 0.25
            
            if not self.IsTransmitting():
                self.SetSignalValue(float(signal_id))
                self.StartTransmitting()

            if self.IsTransmitting():
                if near_prey and signal_id==2:
                    self.transmitted_correctly = True
                    self.times_transmitted += 1
                elif not near_prey or signal_id!=2:
                    self.transmitted_poorly = True
                    self.times_transmitted_false += 1
        else:
            if self.IsTransmitting():
                self.SetSignalValue(0.0)
                self.StopTransmitting()

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
        
        if isinstance(other, EvoBat):
            # Increment the prey captured count
            self.preyCaptured += 1

    def GetFitness(self) -> float:

        n_tx = self.times_transmitted + self.times_transmitted_false

        R_correct = 0.01 * (self.times_transmitted)

        P_wrong = -0.1 * self.times_transmitted_false

        c_tx = -0.01 * (n_tx)

        total_fitness = (R_correct + P_wrong + c_tx + self.preyCaptured + 1)
        print(total_fitness)

        return max(1.0, total_fitness) if self.transmitted_correctly else 1.0
        

class TargetObject(WorldObject):
    """
    A central target object that agents will interact with.
    """

    def __init__(self):
        super().__init__()  # Larger radius to make it easier to hit
        self.SetResetRandom(True)
        self.SetRadius(35.0)
        self.SetColour(0.9, 0.1, 0.1, 1.0)  # Bright red

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
        pass


class EvoBat(EvoFFNAnimat):
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

        # Track whether we're touching the target and times touched
        self.touching_target = False
        self.times_transmitted_pred = 0
        self.times_transmitted_food = 0
        self.times_transmitted_false = 0
        self.transmitted_food_correctly = False
        self.transmitted_pred_correctly = False
        self.transmitted_poorly = False
        self.foodFound = False
        self.times_caught = 0
        self.foodBonus = 0.0

        self.AddSensor(
            "predR", ProximitySensor(Predator, np.pi / 2, 80, np.pi / 8, simple=True)
        )
        self.AddSensor(
            "predL", ProximitySensor(Predator, np.pi / 2, 80, -np.pi / 8, simple=True)
        )

        self.AddSensor(
            "foodR", ProximitySensor(TargetObject, np.pi / 2, 60, np.pi / 8, simple=True)
        )
        self.AddSensor(
            "foodL", ProximitySensor(TargetObject, np.pi / 2, 60, -np.pi / 8, simple=True)
        )

        self.AddSensor("signal", SignalSensor())
        self.SetInteractionRange(200.0)

        self.AddFFNBrain(hidden=10, inputs=4+vocabSize, outputs=2+vocabSize)

        # Set a cyan color for the agent
        Drawable.SetColour(self, 0.0, 0.7, 1.0, 1.0)
    
    def Control(self):
        """
        Control method that handles random movement and signal emission.
        """
        #super().Control()

        outputs = self.GetBrainOutput()

        self.controls["left"] = outputs[0]
        self.controls["right"] = outputs[1]
        
        signals = outputs[2:2+self.vocabSize]
        signal_id = self.GetSelfSignal(signals)

        if (sum(signals)/len(signals)) > 0.0:
            sensors = self.GetSensors()
            pred_sensor_r = sensors["predR"].GetOutput()
            pred_sensor_l = sensors["predL"].GetOutput()
            near_pred = pred_sensor_r > 0.25 or pred_sensor_l > 0.25

            food_sensor_r = sensors["foodR"].GetOutput()
            food_sensor_l = sensors["foodL"].GetOutput()
            near_food = food_sensor_r > 0.3 or food_sensor_l > 0.3
            
            if not self.IsTransmitting():
                self.SetSignalValue(float(signal_id))
                self.StartTransmitting()

            if self.IsTransmitting():
                if near_pred and signal_id==3:
                    self.times_transmitted_pred += 1
                    self.transmitted_pred_correctly = True
                elif near_food and signal_id==1:
                    self.transmitted_food_correctly = True
                    self.times_transmitted_food += 1
                elif not near_pred or not near_food or signal_id==2:
                    self.transmitted_poorly = True
                    self.times_transmitted_false += 1
        else:
            if self.IsTransmitting():
                self.SetSignalValue(0.0)
                self.StopTransmitting()

    
    def Reset(self):
        self.times_transmitted_pred = 0
        self.times_transmitted_food = 0
        self.times_transmitted_false = 0
        self.times_caught = 0
        self.transmitted_food_correctly = False
        self.transmitted_pred_correctly = False
        self.transmitted_poorly = False
        self.foodFound = False
        if self.IsTransmitting():
            self.StopTransmitting()
        super().Reset()

    def OnCollision(self, other):
        # Check if the object we collided with is our target
        if isinstance(other, TargetObject):
            # Mark that we're touching the target this frame
            self.touching_target = True
            self.foodFound = True
            factor = other.getAward()
            if factor > 0:
                self.foodBonus += 0.1 * (2 ** (factor - 1))

            # Start transmitting if not already
            if not self.IsTransmitting():
                pass
                
                # Increment the transmission count for fitness
                #self.times_transmitted += 1
                #self.SetSignalValue(1.0)
                #self.StartTransmitting()
        
        if isinstance(other, Predator):
            self.times_caught += 1
            self.location = self.myWorld.RandomLocation()

    def GetFitness(self) -> float:

        # Temporary fitness to judge how many times bats have transmitted
        individual_fitness = 5.0 if self.foodFound else 0.0
        food_factor = self.foodBonus
        
        n_tx = self.times_transmitted_pred + self.times_transmitted_food + self.times_transmitted_false

        R_correct = 0.01 * (self.times_transmitted_pred + self.times_transmitted_food)

        P_wrong = -0.1 * self.times_transmitted_false

        c_tx = -0.01 * (n_tx)

        total_fitness = (food_factor + individual_fitness + R_correct+P_wrong+c_tx)/(self.times_caught+1)

        return max(1.0, total_fitness) if self.transmitted_food_correctly or self.transmitted_pred_correctly else 1.0
        # return (max(1.0, (3*self.times_transmitted_pred + 1*self.times_transmitted_food -
        #         0.3*self.times_transmitted_false - 15*self.times_caught)))
    
        #return self.times_transmitted/(self.times_caught+1) if self.times_caught > 0 else 1


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

        self.theGA = GeneticAlgorithm( 0.40,0.25, selection = GASelectionType.GA_ROULETTE)
        self.theGA.SetSelection(GASelectionType.GA_ROULETTE)

        self.gaPred = GeneticAlgorithm(0.40, 0.25, selection = GASelectionType.GA_ROULETTE)
        self.gaPred.SetSelection(GASelectionType.GA_ROULETTE)

        thePreds = Population(6, Predator, self.gaPred)        
        theBats =  Population(18, EvoBat, self.theGA)
        theTarget = Group(4, TargetObject)

        self.Add("theBats", theBats)
        self.Add("theTarget", theTarget)
        self.Add("thePreds", thePreds)

        # Logging settings
        self.sleepBetweenLogs = 0.0
        for k in ["Simulation", "Run", "Generation"]:
            self.whatToLog[k] = True
        self.whatToSave["Simulation"] = self.whatToSave["Run"] = self.whatToSave[
            "Generation"
        ] = True

    def LogEndGeneration(self):
        super().LogEndGeneration()

        avg_fitness_prey = np.mean(self.contents["theBats"].AverageFitnessScoreOfMembers())
        max_fitness_prey = np.max(self.contents["theBats"].AverageFitnessScoreOfMembers())

        avg_fitness_pred = np.mean(self.contents["thePreds"].AverageFitnessScoreOfMembers())
        max_fitness_pred = np.max(self.contents["thePreds"].AverageFitnessScoreOfMembers())

        self.logger.info(f"Average fitness prey: {avg_fitness_prey:.2f}, Max fitness prey: {max_fitness_prey:.2f}," 
              f"\nAverage fitness pred: {avg_fitness_pred:.2f}, Max fitness pred: {max_fitness_pred:.2f}, "
        )

        # save the generation, max and average fitness to a log.csv file
        with open("other/log_pred_f.csv", "a") as f:
            f.write(f"{self.generation+1},{avg_fitness_prey},{max_fitness_prey},{avg_fitness_pred},{max_fitness_pred}\n")

    def CreateDataStructSimulation(self):
        self.data = {}

    def CreateDataStructRun(self):
        self.averageFitness = []
        self.maxFitness = []

    def SaveGeneration(self):
        avg_fitness_prey = np.mean(self.contents["theBats"].AverageFitnessScoreOfMembers())
        max_fitness_prey = np.max(self.contents["theBats"].AverageFitnessScoreOfMembers())

        self.averageFitness.append(avg_fitness_prey)
        self.maxFitness.append(max_fitness_prey)

    def SaveRun(self):
        self.data[f"Run{self.Run}_avg"] = self.averageFitness
        self.data[f"Run{self.Run}_max"] = self.maxFitness

# For standalone testing
if __name__ == "__main__":
    simulation = BatBaseSimulation()
    simulation.RunSimulation(render=True)
