import random
from deap import creator, base, tools, algorithms
import ns
import maze
import copy
import numpy as np


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


def clean_debug_states():
    global playout_count
    global behv_state
    global behv_last_visit
    global behv_rewards
    playout_count = 0
    behv_state = copy.deepcopy(env)
    behv_last_visit = copy.deepcopy(env)
    behv_rewards = copy.deepcopy(env)


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


def evalOneMax(individual):
    global playout_count
    global dummy_count
    run_env = copy.deepcopy(env)
    val_reward=0
    for act in individual:
        _, val_reward, done = run_env.step(act)
        if done:
            break

    behv = (run_env.posx, run_env.posy)
    if(behv[0] == 0 and behv[1] == 0):
        r = 0
    else:
        r = n_s.get_approx_novelty(behv, k=5, done=True)
        # r=n_s.get_novelty_simple(behv,done=True)
        r = r*10
    #n_s.set_behavior_in_archive(behv, n_s.behavior_archive, True)
    playout_count += 1
    debug(r, playout_count, run_env)

    return val_reward,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=10)

population = toolbox.population(n=100)

NGEN = 100
i = 0


hof = tools.ParetoFront() # a ParetoFront may be used to retrieve the best non dominated individuals of the evolution
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

algorithms.eaSimple(population, toolbox, 0.7, 0.2, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)
print_debug_states()

"""
for gen in range(NGEN):
    i += 1
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.07)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    print("################ Gen ", i, ' ################')
    print_debug_states()
    clean_debug_states()
"""

top10 = tools.selBest(population, k=10)


for i in range(len(top10)):
    print('it ', i, ': ', top10[i])
    print()
