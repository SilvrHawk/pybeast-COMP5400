"""
The basic Animat comes with no control system, so in the course of deriving
a new type of Animat, a control system must be added. Two useful control
systems are the FeedForwardNet and DynamicalNet classes. FFNAnimat and
DNNAnimat provide Animats with these control systems built-in and
automatically configured from the Animat's sensors and controls. Two other
classes, EvoFFNAnimat and EvoDNNAnimat are provided as evolvable versions
in case the only data contained in the genotype is the ANN configuration.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np

from pybeast.core.agents.animat import Animat
from pybeast.core.control.feedforwardnet import FeedForwardNet
from pybeast.core.control.dynamicalnet import DynamicalNet
from pybeast.core.evolve.evolver import Evolver
from pybeast.core.sensors.sensor import SignalSensor

class BrainAnimat(Animat):

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

        self.myBrain = None
        self.ownBrain = False

    def __del__(self):
        """
        Destructor for DNNAnimat, if the DynamicalNet has been initialised,
        it is deleted here.
        """
        if self.myBrain is not None:
            del self.myBrain
            self.ownBrain = False

    def GetBrain(self):
        """
        Get the brain of the animat.
        """
        return self.myBrain

    def SetBrain(self, brain):
        """
        Sets the brain of the FFNAnimat.
        """

        self.myBrain = brain
    def OwnsBrain(self):
        """
        Check if the animat has its own brain.
        """
        return self.ownBrain


class FFNAnimat(BrainAnimat):
    """
    An Animat with a built-in feed-forward network which is automatically
    configured depending on the Animat's sensor and control configuration.
    See EvoFFNAnimat for an evolvable version.
    """

    def __init__(self):

        super().__init__()

    def AddFFNBrain(self, hidden: int =-1, inputs: int =-1, outputs: int =-1, bias = True) -> None:
        """
        This method is responsible for initialising the FFNAnimat's neural network.
        It should usually be called in the constructor of a derived class, after
        the sensors have been set up. Also randomises the neural network for use in
        evolutionary simulations.

        :param hidden: The number of hidden nodes, defaults to be the same as the
                       number of inputs.
        :param inputs: The number of inputs, defaults to be the same as the number
                       of sensors on the Animat.
        :param outputs: The number of outputs, defaults to be the same as the number
                        of controls on the Animat.
        """

        ## TEMPORARY MEASURES TO COPE WITH MULTIPLE SIGNAL INPUTS
        if hidden == -1:
            hidden = len(self.GetSensors()) + (self.vocabSize -1 if self.vocabSize > 0 else 0)
        if inputs == -1:
            inputs = len(self.GetSensors()) + (self.vocabSize -1 if self.vocabSize > 0 else 0)
        if outputs == -1:
            outputs = len(self.GetControls())

        self.myBrain = FeedForwardNet(inputs, outputs, hidden, bias=bias)
        self.myBrain.Randomise()
        self.ownBrain = True


    def GetBrainOutput(self):
        """Process sensor inputs for the neural network."""
        n = 0
        for sensor in self.GetSensors().values():
            ## OLD METHOD
            # if isinstance(sensor, SignalSensor):
            #     signal_outputs = sensor.GetOutput()
            #     for value in signal_outputs:
            #         self.myBrain.SetInput(n, float(value))
            #         n += 1

            if isinstance(sensor, SignalSensor):
                for signal, avg_strength in self.GetAveragedSignals().items():
                    self.myBrain.SetInput(n, float(avg_strength))
                    n += 1
            else:
                self.myBrain.SetInput(n, float(sensor.GetOutput()))
                n += 1

        self.myBrain.Fire()
        
        return self.myBrain.GetOutputs()

    def Control(self):
        """
        The FFNAnimat's neural net is linked to its sensors and controls here.
        All sensor inputs are fed to the neural network and all control outputs
        are taken from the ANN's output values.

        :warning: It is assumed that there are at least as many input nodes as
                  sensors and at least as many output nodes as controls. If your
                  Animat is not set up in this way your needs are likely greater
                  than can be provided for by FFNAnimat.
        """

        outputs = self.GetBrainOutput()

        for controlName, output in zip(self.controls.keys(), outputs):
            self.controls[controlName] = output

    #================================================================================================
    # Serialise
    #================================================================================================

    def Serialise(self, out):
        """
        Outputs the FFNAnimat's data to a stream.
        :param out: A reference to an output stream.
        """
        # TODO Implement Serialise in a consistent manner througout the project
        assert False, 'Not implemented yet'

        # out.write("FFNAnimat\n")
        # super().Serialise(out)
        # out.write(str(self.myBrain))

    def Unserialise(self, inp):
        """
        Inputs the FFNAnimat's data from a stream.
        :param input: A reference to an input stream.
        """
        # TODO Implement Unserialise in a consistent manner througout the project
        assert False, 'Not implemented yet'

        # name = input.readline().strip()
        # if name != "FFNAnimat":
        #     raise SerialException(SERIAL_ERROR_WRONG_TYPE, name,
        #                           "This object is type FFNAnimat")
        #
        # super().Unserialise(input)
        #
        # if self.myBrain is not None:
        #     del self.myBrain
        # self.myBrain = FeedForwardNet(0, 0)
        # self.myBrain.Unserialise(input)

class EvoFFNAnimat(FFNAnimat, Evolver):
    """
    An evolvable version of FFNAnimat with GetGenotype/SetGenotype methods
    already set up.

    See Also:
        FFNAnimat
    """

    def __init__(self):
        FFNAnimat.__init__(self)
        Evolver.__init__(self)


    def SetGenotype(self, genome: List[float]):
        """
        Here, we receive a genome, i.e. a single list of all parameters and convert it into a config dict which can
        be passed to FFN
        """
        assert len(genome) == (self.myBrain.numberWeights), \
            'A genome must have as many parameters as there are weights in the network'

        inputToHiddenWeights = genome[:self.myBrain.numberInputToHiddenWeights].reshape(
            (self.myBrain.hidden, self.myBrain.inputs + (1 if self.myBrain.biasNode else 0))
        )

        hiddenTooutputWeights = genome[self.myBrain.numberInputToHiddenWeights:].reshape(
            (self.myBrain.outputs, self.myBrain.hidden + (1 if self.myBrain.biasNode else 0))
        )
        config = {'hidden': inputToHiddenWeights, 'output': hiddenTooutputWeights}
        self.myBrain.SetConfiguration(config)


    def GetGenotype(self) -> List[float]:
        """
        As a genome, i.e. the parameters that are trained by the genetic algorithm, we choose the weights of the
        FFN that represents the brain of the animat. All parameters need to be converted into a single array of
        floats so that the gentic algorithm can do is job
        """
        config = self.myBrain.GetConfiguration()
        # weihgts from input to hidden layer
        inputToHiddenWeights = np.array(config['hidden']).flatten()
        hiddenToOutputWeights = np.array(config['output']).flatten()

        return np.concatenate((inputToHiddenWeights, hiddenToOutputWeights))

class DNNAnimat(BrainAnimat):
    """
    An Animat with a built-in dynamical network which is automatically
    configured depending on the Animat's sensor and control configuration.
    See EvoDNNAnimat for an evolvable version.
    TODO: Review brain ownership/destructor
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def InitDNN(self, total=-1, inputs=-1, outputs=-1, multiInput=True, multiOutput=False) -> None:
        """
        Initialize the dynamical network.
        :param total: The total number of nodes, defaults to the number of sensors.
        :param inputs: The number of input nodes, defaults to the number of sensors.
        :param outputs: The number of output nodes, defaults to the number of controls.
        :param multiInput: Whether to allow multiple inputs per node, defaults to True.
        :param multiOutput: Whether to allow multiple outputs per node, defaults to False.
        """
        if total == -1:
            total = len(self.GetSensors()) 
        if inputs == -1:
            inputs = len(self.GetSensors()) + (self.vocabSize -1 if self.vocabSize > 0 else 0)
        if outputs == -1:
            outputs = len(self.GetControls())

        self.myBrain = DynamicalNet(inputs, outputs, total, multiInput, multiOutput)
        self.myBrain.Randomize()
    
    def GetBrainOutput(self):
        """Process sensor inputs for the neural network."""
        n = 0
        for sensor in self.GetSensors().values():
            ## OLD METHOD
            # if isinstance(sensor, SignalSensor):
            #     signal_outputs = sensor.GetOutput()
            #     for value in signal_outputs:
            #         self.myBrain.SetInput(n, float(value))
            #         n += 1
            
            if isinstance(sensor, SignalSensor):
                for signal, avg_strength in self.GetAveragedSignals().items():
                    self.myBrain.SetInput(n, float(avg_strength))
                    n += 1
            else:
                self.myBrain.SetInput(n, float(sensor.GetOutput()))
                n += 1

        self.myBrain.Fire()
        
        return self.myBrain.GetOutputs()

    def Control(self):
        """
        The DNNAnimat's neural net is linked to its sensors and controls here.
        All sensor inputs are fed to the neural network and all control outputs
        are taken from the ANN's output values.
        :warning: It is assumed that there are at least as many input channels as
                  sensors and at least as many output channels as controls. If your
                  Animat is not set up in this way your needs are likely greater
                  than can be provided for by DNNAnimat.
        """
        # n = 0
        # for sensor in self.GetSensors().values():
        #     if isinstance(sensor, SignalSensor):
        #         signal_outputs = sensor.GetOutput()
        #         for value in signal_outputs:
        #             self.myBrain.SetInput(n, float(value))
        #             n += 1
        #     else:
        #         self.myBrain.SetInput(n, float(sensor.GetOutput()))
        #         n += 1

        # self.myBrain.Fire()

        # n = 0
        # for controlName in self.GetControls().keys():
        #     self.GetControls()[controlName] = self.myBrain.GetOutput(n)
        #     n += 1

        # super().Control()
        
        outputs = self.GetBrainOutput()

        for controlName, output in zip(self.controls.keys(), outputs):
            self.controls[controlName] = output

    def Serialise(self, out):
        """
        Outputs the DNNAnimat's data to a stream.
        """

        # TODO: Implement Serialse in a consistent manner throughout the project
        assert False, 'Not implemented yet'

        # out.write("DNNAnimat\n")
        # self.Serialise(out)
        # out.write(str(self.myBrain))

    def Unserialise(self, inp):
        """
        Inputs the DNNAnimat's data from a stream.
        """

        # TODO: Implement Unserialse in a consistent manner throughout the project
        assert False, 'Not implemented yet'

        # name = in_stream.readline().strip()
        # if name != "DNNAnimat":
        #     raise SerialException(SERIAL_ERROR_WRONG_TYPE, name, "This object is type DNNAnimat")
        #
        # self.Unserialise(in_stream)
        #
        # if self.myBrain is not None:
        #     del self.myBrain
        # self.myBrain = DynamicalNet(0, 0, 0)
        # self.myBrain.Unserialise(in_stream)


class EvoDNNAnimat(DNNAnimat, Evolver):
    """
    An evolvable version of DNNAnimat with GetGenotype/SetGenotype methods
    already set up.

    See Also:
        DNNAnimat
    """
    def __init__(self):
        DNNAnimat.__init__(self)
        Evolver.__init__(self)

    def SetGenotype(self, genome: List[float]):
        """
        Convert flat genome list back into the configuration structure needed by DynamicalNet
        """
        # First determine how to split the genome for each neuron
        configs = []
        brain = self.GetBrain()
        neurons = brain.GetNeurons()
        
        # Track our position in the genome
        pos = 0
        
        # For each neuron in the network
        for neuron in neurons:
            input_size = len(neuron.inputWeights)
            output_size = len(neuron.outputWeights)
            weights_size = len(neuron.weights)
            
            # Extract the sections for this neuron
            input_weights = genome[pos:pos+input_size]
            pos += input_size
            
            output_weights = genome[pos:pos+output_size]
            pos += output_size
            
            weights = genome[pos:pos+weights_size]
            pos += weights_size
            
            # Create a configuration dictionary for this neuron
            config = {
                'inputWeights': np.array(input_weights),
                'outputWeights': np.array(output_weights),
                'weights': np.array(weights)
            }
            configs.append(config)
        
        # Set the brain configuration with our restructured configs
        brain.SetConfiguration(configs)

    def GetGenotype(self) -> List[float]:
        """
        Flatten the brain configuration into a single list of floats for the GA
        """
        brain = self.GetBrain()
        configs = brain.GetConfiguration()
        
        # Create a flat list from all the neuron configurations
        flat_genome = []
        for config in configs:
            # Add all inputWeights
            flat_genome.extend(config['inputWeights'].tolist())
            # Add all outputWeights
            flat_genome.extend(config['outputWeights'].tolist())
            # Add all weights
            flat_genome.extend(config['weights'].tolist())
        
        return flat_genome

