import random
import ns
import maze
import copy
import numpy as np
import math

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.algorithm.singleobjective.local_search import LocalSearch
from jmetal.algorithm.singleobjective.simulated_annealing import SimulatedAnnealing


from jmetal.operator import BitFlipMutation, SPXCrossover, BinaryTournamentSelection,SimpleRandomMutation
from jmetal.problem import OneMax
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.core.problem import BinaryProblem, FloatProblem
from jmetal.core.solution import BinarySolution

env = maze.Maze(1)
n_s = ns.NoveltySearch(behavior_type='ad_hoc')
playout_count = 0
dummy_count = 0
behv_state = copy.deepcopy(env)
behv_last_visit = copy.deepcopy(env)
behv_rewards = copy.deepcopy(env)
behv_acc_visit = copy.deepcopy(env)


def print_debug_states():

    global behv_state
    global behv_last_visit
    global behv_rewards
    global behv_acc_visit
    print('accumulative end positions of generation:')
    print(behv_state.render())

    print('accumulative end positions overall')
    print(behv_acc_visit.render())

    print('last visited places')
    print(behv_last_visit.render())

    print('last rewards gained')
    print(behv_rewards.render())



def debug(r, p, state):
    global playout_count
    global behv_state
    global behv_last_visit
    global behv_rewards
    global behv_acc_visit
    behv_state.maze[state.posx, state.posy] += 1
    behv_acc_visit.maze[state.posx, state.posy] += 1
    behv_last_visit.maze[state.posx, state.posy] = p
    behv_rewards.maze[state.posx, state.posy] = r



import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import BinarySolution, Solution, FloatSolution, IntegerSolution, PermutationSolution, \
    CompositeSolution
from jmetal.util.ckecking import Check

class NFlipMutation(Mutation[BinarySolution]):

    def __init__(self, probability: float):
        super(NFlipMutation, self).__init__(probability=probability)

    def execute(self, solution: BinarySolution) -> BinarySolution:
        Check.that(type(solution) is BinarySolution, "Solution type invalid")

        for i in range(solution.number_of_variables):
            for j in range(len(solution.variables[i])):
                rand = random.random()
                if rand <= self.probability:
                    r = random.randint(0,3)
                    solution.variables[i][j] = r
        return solution

    def get_name(self):
        return 'BitFlip mutation'


class evalOneMax(BinaryProblem):
    def __init__(self, number_of_bits: int = 256):
        super(evalOneMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 1
        self.number_of_variables = 3
        self.number_of_constraints = 0

        #self.obj_directions = [self.MAXIMIZE]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Ones']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        global playout_count
        global dummy_count
        run_env = copy.deepcopy(env)
        val_reward=0
        individual=solution.variables[0]

        for act in individual:
            _, val_reward, done = run_env.step(act)
            if done:
                break

        behv = (run_env.posx, run_env.posy)
        #if(behv[0] == 0 and behv[1] == 0):
        #    r = -1
        #else:
        r = n_s.get_approx_novelty(behv, k=5, done=True)
        # r=n_s.get_novelty_simple(behv,done=True)
        r = r*10
        
        n_s.set_behavior_in_archive(behv, n_s.behavior_archive, True)
        playout_count += 1
        debug(r, playout_count, run_env)

        #solution.objectives[0] = val_reward*-1
        solution.objectives[0] = abs(100-r)


        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=3, number_of_objectives=1)
        new_solution.variables[0] = [random.randint(0, self.number_of_variables) for _ in range(self.number_of_bits)]
        return new_solution

    def get_name(self) -> str:
        return 'evalOneMax'



#problem = OneMax(number_of_bits=1024)
problem = evalOneMax(number_of_bits=30)

algorithm = GeneticAlgorithm(
    problem=problem,
    population_size=100,
    offspring_population_size=100,
    #mutation=BitFlipMutation(1.0 / problem.number_of_bits),
    mutation=NFlipMutation(0.1),
    crossover=SPXCrossover(0.8),
    selection=BinaryTournamentSelection(),
    termination_criterion=StoppingByEvaluations(max_evaluations=10000)
)


algorithm.run()
result = algorithm.get_result()

print('Algorithm: {}'.format(algorithm.get_name()))
print('Problem: {}'.format(problem.get_name()))
print('Solution: ' + str(result.variables[0]))
print('Fitness:  ' + str(result.objectives[0]))
print('Computing time: {}'.format(algorithm.total_computing_time))

print_debug_states()
