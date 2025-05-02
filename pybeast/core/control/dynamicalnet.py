'''
file dynamicalnet.h
This file contains the interface for the DynamicalNet object, a
fully-recurrent, continuous-time neural network.
\author David Gordon
biosystems
'''
# Built-in
from typing import List, Tuple
import pickle
# Third-party
import numpy as np

class Neuron:

    def __init__(self,
         N_input: int,
         N_output: int,
         total: int,
         in_ch: int,
         out_ch: int,
         parent: 'DynamicalNet',
         bias: float = 0.0,
         timeConstant: float = 1.0):

        '''
        Constructor: sets the number of inputs, outputs, and weights for
        this neuron, as well as input and output channels.
        :param N_input: The number of input weights.
        :param N_output: The number of output weights.
        :param total: The total number of internal weights.
        :param inCh: The number of input channels.
        :param outCh: The number of output channels.
        :param p: A pointer to the parent DynamicalNet
        '''

        self.inputChannel = in_ch
        self.outputChannel = out_ch

        self.activation = 0.0
        self.inputWeights = np.random.uniform(-1.0, 1.0, N_input)
        self.outputWeights = np.random.uniform(-1.0, 1.0, N_output)
        # self.weights = np.random.uniform(size=total + 2)
        self.weights = np.random.uniform(size=total)

        self.parent = parent
        self.bias = bias
        self.timeConstant = timeConstant

        self.Sigmoid = lambda y: 1.0 / (1.0 + np.exp(-y))

    def __str__(self):
        """
        Prints all the data in this Neuron in a pretty format and returns it as a string.
        :return: A string.
        """
        out = []

        if self.inputWeights:
            out.append("Input weight(s):")
            out.append(self.inputWeights)
            out.append("")

        out.append("Hidden layer weight(s):")
        out.append(self.weights[:-2])
        out.append(f"Bias: {self.bias} Time constant: {self.timeConstant}")

        if self.outputWeights:
            out.append("Output weight(s):")
            out.append(self.outputWeights)
            out.append("")

        return '\n'.join(out)

    def Randomise(self):
        """
        Randomizes the neuron's parameters.
        """
        self.inputWeights[:] = np.random.uniform(-1.0, 1.0, len(self.inputWeights))
        self.outputWeights[:] = np.random.uniform(-1.0, 1.0, len(self.outputWeights))
        self.weights[:-1] = np.random.uniform(-1.0, 1.0, len(self.weights)-1)

        self.bias = self.weights[-2]
        self.timeConstant = np.random.uniform(1.0, 70.0)
        self.weights[-1] = np.log(self.timeConstant)

    def Fire(self):
        """
        This is where everything /actually/ happens - this method calculates the
        amount by which the activation value changes. The code of the Firing method
        has been carefully commented, here is a summary:
        - We start by subtracting the last round's activation.
        - A weighted sum of the previous neuron outputs is taken.
        - Inputs are applied, either as a weighted sum from all channels or as an
        individual input.
        - The total is divided by the time constant...
        - ... and added back to the previous activation.
        - The new activation is then biased and squashed to produce the output.
        - The output is applied to the relevant output channels.
        :return: None
        """
        # Start off with the negative of last round's activation.
        delta_activation = -self.activation
        # Add weighted sum of the other neurons' **output** values
        # (output value = sigmoid(activation - bias)
        delta_activation += np.inner(self.parent.neuronStates, self.weights)

        # Apply any input values
        # ... if there is no particular input channel,
        if self.inputChannel == -1:
            # ... but we have input weights:
            if len(self.inputWeights) > 0:
                # Add a weighted sum of the current inputs and this neuron's
                # input weights.
                delta_activation += np.inner(self.parent.inputs, self.inputWeights)
        # ... if only one input channel goes to this node:
        else:
            # Add the unweighted input value.
            delta_activation += self.parent.inputs[self.inputChannel]

        # Divide by the time constant
        delta_activation /= self.timeConstant

        # And add to the previous activation
        self.activation += delta_activation

        # Bias and squash
        self.output = self.Sigmoid(self.activation - self.bias)

        # Send output values (if this or all neurons are output neurons)
        # ... if there is no particular output channel,
        if self.outputChannel == -1:
            # ... but we have output weights:
            if len(self.outputWeights) > 0:
                # Add the neuron's output to each output channel, weighted by
                # the neuron's output weights.
                self.parent.outputs += self.outputWeights * self.output
        # ... or for just one output neuron:
        else:
            self.parent.outputs[self.outputChannel] += self.output

    def GetOutput(self) -> float:

        return self.output

    def GetConfiguration(self) -> List[float]:
        """
        Copies all weights, bias, and time constant into the provided list.
        :param config: A reference to the list into which the configuration
                       must be copied.
        """

        config = {
            'inputWeights': self.inputWeights,
            'outputWeights': self.outputWeights,
            'weights': self.weights,
        }

        return config

    def SetConfiguration(self, config):
        """
        Sets the Neuron's configuration according to the input iterator, which is an
        iterator of a list of floats. This has been done to enable easy
        configuration by a DynamicalNet of its neurons, without knowing how
        many values are required for each Neuron. The Neuron may therefore
        take what it needs and return an iterator pointing to the rest of the
        configuration data.
        :param config: An iterator pointing to the current position in the
                       configuration data.
        """
        # First come the input weights...
        self.inputWeights = config['inputWeights']
        self.outputWeights = config['outputWeights']
        self.weights = config['weights']
        self.bias = self.weights[-2]
        self.timeConstant = np.exp(self.weights[-1])

        if self.timeConstant < 1.0:
            self.timeConstant = 1.0 + 2 * (1.0 - self.timeConstant)
            self.weights[-1] = np.log(self.timeConstant)

        return

class DynamicalNet:
    '''
    This class implements a fully recurrent continuous (or dynamical) neural
    network. The network is configured with a number of nodes, some or all of
    which may also act as input nodes, and some or all of which may act as
    output nodes. Every node on firing takes a weighted sum of the activation
    states of every other node, including itself. This approach allows the
    network to store information and perform far more complex tasks than a
    feed-forward net might.
    The actual design of dynamical networks has many interpretations, but for
    reference, this one corresponds as closely as possible to the network
    described in Yamauchi, B. M., & Beer, R. D. (1994). Sequential behavior and
    learning in evolved dynamical neural networks. Adaptive Behavior 2(3),
    219--246. http://citeseer.nj.nec.com/yamauchi94sequential.html
    See FeedForwardNet
    '''

    # ------------------------------------------------------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self,
        N_inputs: int,
        N_outputs: int,
        N_total: int,
        multi_input_nodes: bool = True,
        multi_output_nodes: bool = False):

        self.multiInputNodes = multi_input_nodes
        self.multiOutputNodes = multi_output_nodes
        self.inputs = np.zeros(N_inputs, dtype=float)
        self.outputs = np.zeros(N_outputs, dtype=float)
        self.neuronStates = np.zeros(N_total, dtype=float)
        self.neurons = []

        self.Init(N_inputs, N_outputs, N_total, multi_input_nodes, multi_output_nodes)

    def __del__(self):
        pass

    def __str__(self):
        """
        Prints all the data in this network in a pretty format and returns it as a string.
        :return: A string.
        """
        out = []
        out.append("Input values:")
        out.append(self.inputs)
        out.append("Output values:")
        out.append(self.outputs)
        out.append("Activation states:")
        out.append(self.neuronStates)
        out.append("Neurons:")
        out.extend(self.neurons)
        return '\n'.join(out)

    # ------------------------------------------------------------------------------------------------------------------
    # Class interface methods
    # ------------------------------------------------------------------------------------------------------------------

    def Init(self, N_inputs, N_outputs, N_total, mi, mo):

        neuronInputs = N_inputs if mi else 0
        neuronOutputs = N_outputs if mo else 0
        neuronInChl = -1
        neuronOutChl = -1

        for n in range(N_total):
            if not mi:
                neuronInChl = n if n < N_inputs else -1
            if not mo:
                neuronOutChl = n + (N_outputs - N_total)
                if neuronOutChl < 0 or neuronOutChl >= N_outputs:
                    neuronOutChl = -1

            neuron = Neuron(neuronInputs, neuronOutputs, N_total, neuronInChl, neuronOutChl, self)
            self.neurons.append(neuron)

        self.Reset()


    def Reset(self):
        """
        Sets the output value of each neuron (i.e. its current output) to 0.
        This is always done on initialization of the network.
        """
        self.neuronStates.fill(0.0)

    def Randomize(self):
        """
        Forces each Neuron in the network to randomize itself, by calling its own
        Randomise member function.
        :return: None
        """
        for neuron in self.neurons:
            neuron.Randomise()

    def Fire(self):
        """
        This is where it all happens, although all the DynamicalNet class really
        has to do is:
        - Clear the outputs.
        - Fire every Neuron.
        - Retrieve and store every Neuron's output.
        :return: None
        """
        # Clear the output values
        self.outputs[:] = 0.0

        # Call Fire on every neuron
        for neuron in self.neurons:
            neuron.Fire()

        # Store the output value of each neuron for next time.
        for i, neuron in enumerate(self.neurons):
            self.neuronStates[i] = neuron.GetOutput()

    # ------------------------------------------------------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------------------------------------------------------

    def GetOutput(self, n: int) -> float:
        return self.outputs[n]

    def GetOutputs(self) -> List[float]:
        return self.outputs


    def GetConfiguration(self) -> List[float]:
        """
        Returns all the weights and biases in the network, in a long list suitable
        for processing by a GA. Note that nearly all the values are initialized in
        the range [-1,1], except for the time constants which range between 1 and
        70. The time constants are therefore stored as their natural log in the
        configuration output which makes for more sensible alteration by the GA.
        :return: An ordered list of floats describing the weights, biases, and
                 time constants.
        """

        configs = []

        for neuron in self.neurons:
            configs.append(neuron.GetConfiguration())

        return configs

    def GetInputs(self) -> List[float]:
        return self.inputs


    def GetNeuronStates(self) -> List[float]:
        return self.neuronStates

    def GetNeurons(self) -> List[Neuron]:
        return self.neurons


    def IsMultiInputNodes(self) -> bool:
        return self.multiInputNodes

    def IsMultiOutputNodes(self) -> bool:
        return self.multiOutputNodes

    def GetConfigurationLength(self) -> int:
        pass
        # TODO: Not needed anymore

    # ------------------------------------------------------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------------------------------------------------------

    def SetInputChannel(self, neuron: int, channel: int):
        """
        Configures the network to channel inputs to different nodes. Has no effect
        if the net is configured with multiple input nodes per channel.
        @param neuron The number of the node to direct the input to.
        :param: channel The number of the channel to be redirected.
        :return: None
        """
        if self.multiInputNodes or channel < 0 or channel >= len(self.inputs):
            return

        for i in self.neurons:
            if i.inputChannel == channel:
                i.inputWeights.Clear()

        self.neurons[neuron].inputChannel = channel
        self.neurons[neuron].inputWeights = [0.0]

    def SetOutputChannel(self, neuron: int, channel: int):
        """
        Configures the network to channel output from different nodes. Has no
        effect if the net is configured with multiple output nodes per channel.
        :param neuron: The number of the node to be redirected.
        :param channel: The number of the channel to redirect it to.
        :return: None
        """
        if self.multiOutputNodes or channel < 0 or channel >= len(self.outputs):
            return

        for i in self.neurons:
            if i.outputChannel == channel:
                i.outputWeights.Clear()

        self.neurons[neuron].outputChannel = channel
        self.neurons[neuron].outputWeights = [0.0]

    def SetInput(self, n: int, f: float):
        self.inputs[n] = f

    def SetInputs(self, v: List[float]):
        self.inputs = v

    def SetConfiguration(self, configs: List[float]):
        """
        Sets the configuration of the network according to a provided list of
        weights, biases, and time constants.
        :param config: An ordered list of floats containing weights, biases, and
                       time constants.
        """
        for neuron, config in zip(self.neurons, configs):
            neuron.SetConfiguration(config)

    def Serialize(self, out: str):
        """
          Serialize the DynamicalNet object and save it to a file using pickle.
          :param out: The filename to save the serialized object.
          """
        with open(out, 'wb') as f:
            pickle.dump(self, f)


    def Unserialize(self, inp: str):
        """
        Deserialize a DynamicalNet object from the input stream.
        :param inp: The filename to load serialized object.
        """

        with open(inp, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
