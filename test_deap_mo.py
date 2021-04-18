# Create the item dictionary:
import numpy as np 
import numpy
import random
import copy
from numpy import array
from deap import base, creator, tools, algorithms

###  Multi-objective Optimization Problem  ###

IND_INIT_SIZE = 5

MAX_WEIGHT = 2000 # kg
MAX_SIZE = 1500 # m**3

r = array([[213, 508,  22],  # 1st arg : weight / 2nd arg : size / 3rd arg : value
       [594, 354,  50],
       [275, 787,  43],
       [652, 218,  46],
       [728, 183,  43],
       [856, 308,  33],
       [727, 482,  45],
       [762, 683,  26],
       [707, 450,  19],
       [909, 309,  45],
       [979, 247,  42],
       [259, 705,  42],
       [260, 543,  14],
       [899, 825,  17],
       [446, 360,  35],
       [491, 818,  47],
       [647, 404,  17],
       [604, 623,  32],
       [900, 840,  45],
       [374, 127,  33]] )


NBR_ITEMS = r.shape[0]
IND_INIT_SIZE=20

items = {}
# Create random items and store them in the items' dictionary.
for i in range(NBR_ITEMS):
    items[i] = ( r[i][0] , r[i][1] , r[i][2] )


creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0 ))  # Note here <- I used only two weights!  (at first, I tried weights=(-1.0 , -1.0, 1.0)) but it crashes. With deap, you cannot do such a thing.

creator.create("Individual", set, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_item", random.randrange, NBR_ITEMS)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, n=IND_INIT_SIZE) #
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluation(individual):
    weight = 0.0
    size =0.0
    value = 0.0

    # Maximize or Minimize Conditions
    for item in individual:
        weight += items[item][0]  # It must be minimized.
        size += items[item][1]  # It must be minimized.
        value += items[item][2]  # It must be maximized.

    # Limit Conditions
    if weight > MAX_WEIGHT or size > MAX_SIZE:
        return 10000, 0

    if value == 0:
        value = 0.0000001

    MinFitess_score = weight + size   # NOTE : Minimize weight, size
    MaxFitenss_score = value  # NOTE : Maximize weight, size

    return MinFitess_score , MaxFitenss_score,



def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)  # Used in order to keep type
    ind1 &= ind2  # Intersection (inplace)
    ind2 ^= temp  # Symmetric Difference (inplace)
    return ind1, ind2


def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:  # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,  # NOTE comma(,) , if there's no comma, an error occurs.



toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2) # NSGA-2 applies to multi-objective problems such as knapsack problem
toolbox.register("evaluate", evaluation)


def main():
    ngen = 300  # a number of generation  < adjustable value >

    pop = toolbox.population(n= 300)
    hof = tools.ParetoFront() # a ParetoFront may be used to retrieve the best non dominated individuals of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

    return hof, pop



if __name__ == "__main__":
    hof, pop = main()

    print(hof) 