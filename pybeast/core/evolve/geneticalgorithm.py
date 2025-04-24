'''
This collection of classes implement a range of biosystems algorithms and
control systems including a multi-purpose genetic algorithm and two neural
nets. There are also classes for using these controllers in the simulation
environment.
'''

# Built-in
from abc import ABC, abstractmethod
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Union, Optional, TypeVar, TYPE_CHECKING
import csv
import io
import random
# From third-party
import numpy as np
# From pybeast
from pybeast.core.evolve.evolver import Evolver

if TYPE_CHECKING:
    from pybeast.core.world.world import World
    from pybeast.core.evolve.population import Population

Genotype = List[float]
EVO = TypeVar('evolver', bound=Evolver)

class MutationOperator(ABC):
    '''
    A functor which may be initialised with max and min values, and then
    returns a uniformly distributed random number between those values.
    MutationOperator may be adapted in a number of ways to suit different GA
    requirements.txt:
    The template can be initialised as any numeric type and will return good
    results for real values, usable results for integer types (remember to add
    one to the max limit to take account of the lack of proper rounding.)

    The template can be specialised to provide an alternative default mutation
    operator for whichever type you are using for your genotype, e.g. you might
    want to make a gaussian distributed float mutation operator.

    Alternatively you could do away with MutationOperator entirely and specify
    a different functor type when initialising your GA (the GA defaults to a
    mutation operator of a type matching the GA's gene type)
    '''
    @abstractmethod
    def __call__(self):
        pass

class UniformMutator(MutationOperator):

    def __init__(self, minimum = -0.2, maximum = 0.2):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, t: float):
        return t + random.uniform(self.minimum, self.maximum)

class NormalMutator(MutationOperator):
    def __init__(self, mean: float = 0.0, sd: float = 0.1):
        self.mean = mean
        self.sd = sd

    def __call__(self, t):
        return t + random.normalvariate(self.mean, self.sd)

GASelectionType = SimpleNamespace(
    GA_ROULETTE=0,
    GA_RANK=1,
    GA_TOURNAMENT=2
)

GAPrintStyleType = SimpleNamespace(
    GA_PARAMETERS=1,
    GA_CURRENT=2,
    GA_GENERATION=4,
    GA_HISTORY=8
)

GAFitnessMethodType = SimpleNamespace(
    GA_BEST_FITNESS=0,
    GA_WORST_FITNESS=1,
    GA_MEAN_FITNESS=2,
    GA_TOTAL_FITNESS=3
)

GAFitnessFixType = SimpleNamespace(
    GA_IGNORE=0,
    GA_CLAMP=1,
    GA_FIX=2
)

GAFltParamDefault = SimpleNamespace(
    GA_TOURNAMENT_PARAM=0.75,
    GA_RANK_SPRESSURE=1.5,
    GA_EXPONENT=1.0
)

GAIntParamDefault = SimpleNamespace(
    GA_TOURNAMENT_SIZE=5
)

class GeneticAlgorithm(ABC):
    """
    The GeneticAlgorithm class provides functionality to cover a range of GA
    methods, and may be extended to incorporate other approaches. The class is
    completely generic, in order to allow the widest possible application.
    The first template parameter is the type of the genes (e.g. int, float).
    The second is a class which must provide the methods exposed by the Evolver ABC,
    ideally inherited from an Evolver of the same templated type.
    The third template parameter is the type of the mutation operator and will
    usually be fine as the default type of MutationOperator<T>, which is a
    MutationOperator function object with the same type as the genes.
    Phew!
    """

    def __init__(self,
                 crossover: float = 0.7,
                 mutation: float = 0.05,
                 crossoverPoints: int = 1,
                 teamsize: int = -1,
                 selection: int = GASelectionType.GA_RANK,
                 fitnessMethod: int = GAFitnessMethodType.GA_BEST_FITNESS,
                 fitnessFix: int = GAFitnessFixType.GA_IGNORE,
                 printStyle: List[int] = [GAPrintStyleType.GA_PARAMETERS],
                 elitism: int = 0,
                 culling: int = 0,
                 mutFunc: Optional[Union[callable, MutationOperator]] = None):

        assert culling % 2 == 0, "'culling' must be even"

        self.crossover = crossover  # The rate of crossover
        self.mutation = mutation   # The rate of mutation
        self.crossoverPoints = crossoverPoints # Number of crossover points (default 1)
        self.elitism = elitism # Number to go through unchanged
        self.culling = culling  # Number barred from reproduction
        self.teamsize = teamsize # Setting teamsize allows us to assess only subset of the population
        self.outputPopulation = [] # Stores the newly generated population

        self.selection = selection # The selection method (see GASelectionType)
        self.fitnessMethod = fitnessMethod # Selection method for multiple fitnesses
        self.fitnessFix = fitnessFix # What to do with fitness scores < 0

        self.fltParams = GAFltParamDefault # float parameters
        self.intParams = GAIntParamDefault # integer parameters

        # The mutation function, either a unary function object...
        if mutFunc is None:
            self.mutFunc = NormalMutator()  # normal function
        else:
            self.mutFunc = mutFunc

        self.printStyle = printStyle  # The current printing style

        self.ownsData = False # Whether the GA is responsible for deletion
        self.generations = None

        # Data used during creation of a new generation
        self.inputPopSize = 0  # The size of the input population
        self.chromoLength = 0  # Chromosome length, calculated from first individual
        self.totalFitness = 0.0  # The total unscaled fitness of the whole population
        self.bestFitness = 0.0  # The best fitness in the current generation
        self.totalFixedFitness = 0.0  # Total fitness after scaling
        self.worstFitness = 0.0  # Lowest fitness score in the whole population
        self.totalProbability = 0.0  # Total of probability distribution calculation

        # Attributes for storing the algorithm's history
        self.generations = 0  # Number of generations so far
        self.averageFitnessRecord = []  # Average fitness of each generation
        self.bestFitnessRecord = []  # Best fitness of each generation
        self.bestEverFitness = 0.0  # Best fitness score so far
        self.bestEverGenome = None  # A copy of the best candidate ever
        self.bestCurrentGenome = None  # The best candidate of this generation

    def __del__(self):
        """
        Destructor not implemented
        """
        pass

    @property
    def __str__(self):
        """
        Returns a string containing various details about the GA's current state,
        depending on what has been set with SetPrintStyle. Options include:
        GA_PARAMETERS: print the current parameters
        GA_CURRENT: print stats for the current generation
        GA_GENERATION: output the current generation
        GA_HISTORY: print the history of average and best fitness
        """
        # TODO: store time data

        out = io.StringIO()

        if GAPrintStyleType.GA_PARAMETERS in self.printStyle:
            out.write(f"Crossover: {self.crossover:.2f}  Mutation: {self.mutation:.2f}\n")
            out.write("Selection type: ")
            if self.selection == self.GASelectionType.GA_ROULETTE:
                out.write("roulette wheel selection\n")
            elif self.selection == self.GASelectionType.GA_RANK:
                out.write("rank selection\n")
            elif self.selection == self.GASelectionType.GA_TOURNAMENT:
                out.write("tournament selection\n")
                out.write(
                    f"Tournament size: {self.intParams.GA_TOURNAMENT_SIZE} Chance of win: {self.fltParams.GA_TOURNAMENT_PARAM}\n")
            out.write(f"Elitism: {self.elitism}  Sub-elitism: {self.culling}\n")
            out.write(f"Output population size: {self.outputPopSize}\n\n")

        if self.GAPrintStyleType.GA_CURRENT in self.printStyle:
            out.write(
                f"Generation: {self.generations:}   Average fitness: {(self.totalFitness / self.inputPopSize):.2f}   Best fitness: {self.bestFitness:.2f}\n")

        if self.GAPrintStyleType.GA_GENERATION in self.printStyle:
            i = 1
            for evo in self.population.members:
                out.write(f"{i:<3}: ")
                for ch in evo.GetGenotype():
                    out.write(f"{ch} ")
                out.write(f" Fitness: {self.GetFitness(evo.cheesesFound, evo.distanceTravelled) :.2f}\n")
                i += 1
            out.write("\n")

        if self.GAPrintStyleType.GA_HISTORY in self.printStyle:
            out.write("  Gen   |   Avg   |  Best\n")
            for i, (avg, best) in enumerate(zip(self.averageFitnessRecord, self.bestFitnessRecord)):
                out.write(f"{i:<8} {avg:<8.2f}  {best:<8.2f}\n")

        return out.getvalue()

    #================================================================================================
    # Class Interface
    #================================================================================================

    def Generate(self):
        """
        The generation function: this is where it all happens.
        First the total, best, worst and average fitnesses are calculated with
        CalcStats(). Then fitness scaling and probability distributions are
        done by Setup(). Elitism and subelitism are dealt with and then the
        new population is generated using the current selection method,
        crossover parameters and mutation operator.
        """
        # Calculate statistics
        self.CalcStats()
        # Setup fitness scaling and probability distributions
        self.Setup()

        self.outputPopulation.clear()
        # Remove worst individuals if necessary
        if self.culling > 0:
            for i in range(self.culling):
                self.population.members.pop()
        # Fill the new generation with top individuals
        if self.elitism > 0:
            for evo in self.population.members[:self.elitism]:
                newEvo = self.NewMember()
                newEvo.Init()
                newEvo.SetGenotype(evo.GetGenotype())
                self.outputPopulation.append(newEvo)

        # Generate new population
        for i in range((self.outputPopSize - self.elitism) // 2):
            # Select two parents
            geno1 = self.SelectParentGenotype()
            geno2 = self.SelectParentGenotype()

            # cross-over
            for j in range(self.crossoverPoints):
                if random.random() < self.crossover:
                    geno1, geno2 = self.CrossoverGenotypes(geno1, geno2)

            self.MutateGenotype(geno1)
            self.MutateGenotype(geno2)
            evo1 = self.NewMember()
            evo2 = self.NewMember()
            evo1.SetGenotype(geno1)
            evo2.SetGenotype(geno2)
            self.outputPopulation.append(evo1)
            self.outputPopulation.append(evo2)

    def CalcStats(self):

        # First find out how many we're dealing with
        self.inputPopSize = len(self.population.members)

        assert self.inputPopSize % 2 == 0, "'popSize' must be even"
        assert self.culling <= self.inputPopSize - 2, "'culling' must be smaller than 'popSize'-2"
        assert self.elitism <= self.inputPopSize - self.culling, "'elitism' must be smaller than 'popSize' - 'culling'"

        bestEvoSoFar = self.population.members[0]
        self.bestFitness = self.worstFitness = self.GetFitness(bestEvoSoFar)
        self.totalFitness = 0.0

        # Determine best and worst fitness within pop
        for evo in self.population.members:
            f = self.GetFitness(evo)
            # Check if member has been assessed
            if f is None:
                continue
            if f > self.bestFitness:
                bestEvoSoFar = evo
                self.bestFitness = f
            elif f < self.worstFitness:
                self.worstFitness = f

            # Calc pops total fitness
            self.totalFitness += f

        # Store some statistics
        self.generations += 1
        self.averageFitnessRecord.append(self.totalFitness / float(self.inputPopSize))
        self.bestFitnessRecord.append(self.bestFitness)
        self.bestCurrentGenome = bestEvoSoFar.GetGenotype()

        if self.bestFitness > self.bestEverFitness:
            self.bestEverFitness = self.bestFitness
            self.bestEverGenome = self.bestCurrentGenome

    def Setup(self):
        """
        Prepares the GA for the next epoch. Output population is cleared, input population is sorted by fitness,
        population probabilities are set according to the selection method in use, the chromosome length is determined,
        etc.
        """
        self.outputPopulation.clear()
        self.outputPopSize = len(self.population.members)

        # TODO: perhaps make chromoLength the shortest of any given pair?
        self.chromoLength = len(self.population.members[0].GetGenotype())

        # Normalize fitness such that its greater equal zero
        self.FixFitness()

        # Elitism, subelitism and rank selection require population to be sorted by fitness (best-worst)
        self.population.members.sort(key=lambda x: x.GAFixedFitness, reverse=True)

        # Probability distribution Setup:
        # Each individual has a public attribute, GAProbability, which is used
        # to store probability distribution data. In the following code this
        # attribute is precalculated in order to speed up the selection process.
        self.totalProbability = 0

        # Roulette wheel selection:
        # Each individual's GAProbability attribute is primed with a value
        # corresponding to their fitness compared to the overall population.
        if self.selection == GASelectionType.GA_ROULETTE:
            for evo in self.population.members:
                if evo.GAFitness is None:
                    evo.GAProbability = 0.0
                else:
                    evo.GAProbability = (evo.GAFixedFitness / self.totalFixedFitness) ** self.fltParams.GA_EXPONENT
                    #print(self.totalFixedFitness)
                    self.totalProbability += evo.GAProbability

        # Rank selection:
        # Use rank of the individual instead of fitness to prevent overwhelmingly fit
        # individuals to dominate selection
        if self.selection == GASelectionType.GA_RANK:
            for rank, evo in enumerate(self.population.members):
                if evo.GAFitness is None:
                    evo.GAProbability = 0.0
                else:
                    evo.GAProbability = (1.0 - rank / (self.inputPopSize-1)) ** self.fltParams.GA_EXPONENT
                    self.totalProbability += evo.GAProbability

        if self.selection in [GASelectionType.GA_ROULETTE, GASelectionType.GA_RANK]:
            # Normalize propabilities to ensure that they sum to 1
            for evo in self.population.members:
                evo.GAProbability /= self.totalProbability
        return

    def CleanUp(self):
        """
        Deletes the objects comprising the input and output populations.
        """

        if self.ownsData:
            self.population.clear()
            self.outputPopulation.clear()
        else:
            self.population = []
            self.outputPopulation = []

    # ================================================================================================
    # Helper methods to generate members of next generation
    # ================================================================================================

    def NewMember(self) -> EVO:
        """
        Initializes new individual which shares the same class type as members of population
        """
        return self.population.Type(*self.population.args, **self.population.kwargs)

    def FixFitness(self) -> None:
        """
        Normalize fitness according 'GAFitnessFixType' such that it greater than zero
        """
        totalFixedFitness = 0

        for evo in self.population.members:
            f = evo.GAFitness
            if f is None:
                continue
            elif self.fitnessFix == GAFitnessFixType.GA_FIX:
                f -= self.worstFitness
            elif self.fitnessFix == GAFitnessFixType.GA_CLAMP:
                f = max(0, f)
            elif self.fitnessFix == GAFitnessFixType.GA_IGNORE:
                pass

            evo.GAFixedFitness = f
            totalFixedFitness += f

        self.totalFixedFitness = totalFixedFitness

    def SelectParentGenotype(self) -> Genotype:
        """
        Depending on the selection procedure chosen, a genotype is taken from the
        population and returned by reference.
        """
        if self.selection in [GASelectionType.GA_ROULETTE, GASelectionType.GA_RANK]:
            chromo = self.SelectProbability()
        elif self.selection == GASelectionType.GA_TOURNAMENT:
            chromo = self.SelectTournament()
        else:
            assert False, "'self.selection' must be one of 'GASelectionType'"

        return chromo

    def SelectProbability(self) -> Genotype:
        """
        Roulette and Rank Selection
        Having done the Setup function (above), each individual has a
        probability score, derived from their rank or fitness.
        Imagine all the probability scores of the population as a big pie
        chart, printed on a roulette wheel. Now imagine numbers around the
        edge, starting at 0, going up to the total probability of selection.
        We pick a random number between 0 and 1, and call it slice.
        This is where our roulette ball will land.
        """
        probabilities = [evo.GAProbability for evo in self.population.members]
        evo = random.choices(self.population.members, weights=probabilities, k=1)[0]

        return evo.GetGenotype()

    def SelectTournament(self) -> Genotype:
        """
        Tournament Selection
        The method implemented here is an amalgamation of two slightly different
        approaches. GA_TOURNAMENT_SIZE individuals are selected at random from
        the population. With GA_TOURNAMENT_PARAM probability, the fittest
        individual is picked, otherwise a random individual (perhaps still the
        fittest) is chosen from the tournament.
        """
        # Pick GA_TOURNAMENT_SIZE individuals at random
        tournament = random.sample(self.population.members, self.intParams.GA_TOURNAMENT_SIZE)

        # Decide the winner...
        # Depending on GA_TOURNAMENT_PARAM,
        if random.random() < self.fltParams.GA_TOURNAMENT_PARAM:
            # The fittest individual is chosen:
            winner = max(tournament, key=lambda evo: evo.GAFixedFitness)
        else:
            # Or an individual is chosen at random
            winner = random.choice(tournament)

        chromo = winner.GetGenotype()

        return chromo

    def CrossoverGenotypes(self, mum: Genotype, dad: Genotype):
        """
        This function simulates the crossover operation in a genetic algorithm.
        It takes two chromosomes, 'mum' and 'dad', and swaps them over at a
        random point along the length.

        Args:
            mum (list): The first chromosome.
            dad (list): The second chromosome.

        Returns:
            None. The function modifies the input chromosomes in place.
        """
        crossover_point = random.randint(0, self.chromoLength-1)

        child1 = np.concatenate((mum[:crossover_point], dad[crossover_point:]))
        child2 = np.concatenate((dad[:crossover_point], mum[crossover_point:]))

        return child1, child2

    def MutateGenotype(self, genom: Genotype):
        """
        While crossover simulates the effect of sexual reproduction within a
        population, mutation artificially reproduces the effects of transcription
        errors in the replication of DNA.
        """
        for i in range(len(genom)):
            if random.random() < self.mutation:
                genom[i] = self.mutFunc(genom[i])

    def GetFitness(self, evo: EVO) -> float:
        """
        Calculates the fitness score to be used by the GA from the stored fitness
        scores in the Evo object. If no scores have been stored, the output from
        EVO::GetFitness is returned.
        """
        if not evo.GAFitnessScores:
            return None

        if self.fitnessMethod == GAFitnessMethodType.GA_BEST_FITNESS:
            f = np.max(evo.GAFitnessScores)
        elif self.fitnessMethod == GAFitnessMethodType.GA_WORST_FITNESS:
            f = np.min(evo.GAFitnessScores)
        elif self.fitnessMethod == GAFitnessMethodType.GA_MEAN_FITNESS:
            f = np.mean(evo.GAFitnessScores)
        elif self.fitnessMethod == GAFitnessMethodType.GA_TOTAL_FITNESS:
            f = np.sum(evo.GAFitnessScores)
        else:
            f = 0.0

        evo.GAFitness = f

        return f

    def CalcFitness(self, evo):

        """
        Calculates the fitness score to be used by the GA from the stored fitness
        scores in the Evo object. If no scores have been stored, the output from
        EVO::GetFitness is returned.
        """
        if not evo.GAFitnessScores:
            return GetFitness(evo.cheesesFound, evo.distanceTravelled)
        if self.fitnessMethod == GAFitnessMethodType.GA_BEST_FITNESS:
            return np.max(evo.GAFitnessScores)
        elif self.fitnessMethod == GAFitnessMethodType.GA_WORST_FITNESS:
            return np.min(evo.GAFitnessScores)
        elif self.fitnessMethod == GAFitnessMethodType.GA_MEAN_FITNESS:
            return np.mean(evo.GAFitnessScores)
        elif self.fitnessMethod == GAFitnessMethodType.GA_TOTAL_FITNESS:
            return np.sum(evo.GAFitnessScores)
        else:
            return 0.0

    #================================================================================================
    # Getter and Setter
    #================================================================================================

    def GetPopulation(self) -> List[EVO]:
        return self.outputPopulation

    def GetPopulationCopy(self) -> List[EVO]:
        """
        Returns duplicates of the output population which won't be deleted by the GA if it does a CleanUp().
        """
        return deepcopy(self.outputPopulation)

    def GetAvgFitnessHistory(self) -> List[float]:
        return self.bestFitnessRecord

    def GetBestCurrentGenome(self) -> Genotype:
        return self.bestCurrentGenome

    def GetBestEverGenome(self) -> Genotype:
        return self.bestEverGenome

    def GetBestCurrentFitness(self) -> float:
        return self.bestFitness

    def GetBestEverFitness(self) -> float:
        return self.bestEverFitness

    def SetPopulation(self, p: "Population"):
        self.population = p




    def SetCrossover(self, c: float):
        """
        Set crossover rate
        """
        assert 0 <= c <= 1.0, "Crossover rate 'c' must be in the range [0, 1]"

        self.crossover = c

    def SetMutation(self, m: float):
        """
        Sets the rate of mutation.
        """
        assert 0 <= m <= 1.0, "Mutation rate 'm' must be in the range [0, 1]"

        self.mutation = m

    def SetSelection(self, s: GASelectionType):
        """
        Sets the selection namespace.

        :param s: The selection namespace
        """
        self.selection = s

    def SetMutationFunction(self, func: Union[callable, MutationOperator]):
        """
        Sets the mutation function.

        The mutation function may be any function which takes as its argument
        one variable of the gene type and returns a variable of the gene type.

        :param func: A function which takes a variable of the gene type and returns a variable of the gene type.
        """
        self.mutFuncObj = func

    def SetElitism(self, e: int):
        """
        Sets the elitism value, which decides how many of the fittest individuals
        go through to the next generation unchanged.

        :param e: An integer in the range [0, population size].
        """
        assert 0 <= e <= self.popSize, "Variable 'e' must be in the range [0, popSize]"
        self.elitism = e

    def SetSubelitism(self, s: int):
        """
        Sets the subelitism value, which decides how many of the least fit
        individuals are barred from reproducing.

        :param s: An integer in the range [0, population size].
        """
        assert 0 <= s <= self.popSize, "Variable 'e' must be in the range [0, popSize]"
        self.culling = s

    def SetCrossoverPoints(self, p: int):
        """
        Sets the number of points of crossover.


        :param p: An integer >= 0 (0 results in no crossover).
        """
        assert p >= 0, "Crossover points must be a non-negative integer"

        self.crossoverPoints = p

    def SetFitnessMethod(self, f: str):
        """
        Sets the method by which fitness scores are used.

        :param f: Choose method from 'GAFitnessMethodType'
        """
        assert f in self.GAFitnessMethodType, f" 'f' must be in {self.GAFitnessMethodType.__dict__.keys()}"

        self.fitnessMethod = getattr(self.GAFitnessMethodType, f)

    def SetFitnessFix(self, f: str):
        """
	    Sets the method by which fitness scores < 0 are treated.

        :param f: Choose method from 'GAFitnessFixType'
        """
        assert f in self.GAFitnessFixType, f" 'f' must be in {self.GAFitnessFixType.__dict__.keys()}"
        self.fitnessFix = getattr(self.GAFitnessFixType, f)

    def SetOwnsData(self, b: bool):
        """
        Set to True if the GA is responsible for deleting the old population objects.

        :param b: True if the GA is responsible, False otherwise.
        """

        self.ownsData = b

    def SetIntParameter(self, a: str, v: int):
        """
        Set to True if the GA is responsible for deleting the old population objects.

        :param p: Attribute to set
        :param v: Value
        """

        setattr(self.intParams, a, v)

    def SetFltParameter(self, a: str, v: float):
        """
        Set to True if the GA is responsible for deleting the old population objects.

        :param p: Attribute to set
        :param v: Value
        """

        setattr(self.fltParams, a, v)

    def SetPrintStyle(self, p: int):
        self.printStyle = p

    #================================================================================================
    # Serialiase
    #================================================================================================

    def GetCSV(self, separator: str = ','):
        """
        Returns a string containing a simple CSV table with average and best
        fitness for every generation so far. The default separator is "," but
        a different one may be specified.
        """
        output = io.StringIO()
        writer = csv.writer(output, delimiter=separator)
        writer.writerow(['Average fitness', 'Best fitness'])
        for avg, best in zip(self.averageFitnessRecord, self.bestFitnessRecord):
            writer.writerow([avg, best])

        return output.getvalue()

    @staticmethod
    def Unserialise(inp: str) -> "GeneticAlgorithm":
        # TODO: Implement in a consistent manner throughout the project
        assert False, 'Not implemented yet'

    def Serialise(self, out: str):
        """
        Writes the serialized data of the GeneticAlgorithm object to the output stream.

        Parameters:
            out (str): Output filename

        """
        # TODO: Implement in a consistent manner throughout the project
        assert False, 'Not implemented yet'
