import numpy as np
import sys
import time
import random
import copy
import maze
import player
env = maze.Maze(1)
env.render()

def run():
    #Monte Carlo Tree Search
    #acts =player.mcts_act(env,1000,15,do_ns=False)
    #Mutation Hill Climbing
    #acts =player.rhea_act(env,it=1,pop_evolutions=100,pop_num=1,rollout_limit=30,mutation_prob = 0.2,
    #                      do_ns=False,run_type=3)
    #RHEA
    #acts =player.rhea_act(env,it=1,pop_evolutions=200,pop_num=12,rollout_limit=40,mutation_prob = 0.2,
    #                      do_ns=True,run_type=3)
    
    #acts =player.rhea_act(env,it=1,pop_evolutions=300,pop_num=4,rollout_limit=40,mutation_prob = 0.2,
    #                      do_ns=True,run_type=3)

    #print("Finished Planning!!!")
    #print("Best plan:",acts)    


    #Others- Experimental
    #acts =player.cmcts_act(env,100,20,do_ns=True,sel_type='rave_novelty',update_type ='amaf')
    acts =player.cmcts_act(env,300,50,do_ns=True,sel_type='avg_novelty')
    #acts =player.rhea_act(env,it=300,pop_evolutions=5,pop_num=5,rollout_limit=15,mutation_prob = 0.2,
    #                      do_ns=True,run_type=3)
    
    #exp
    #acts =player.cmcts_act(env,300,30,do_ns=True,sel_type='avg_novelty')
    #acts =player.rhea_act(env,it=1,pop_evolutions=100,pop_num=10,rollout_limit=30,mutation_prob = 0.07,
    #                      do_ns=True,run_type=3)
    #acts =player.rhea_act(env,it=1,pop_evolutions=2000,pop_num=1,rollout_limit=30,mutation_prob = 0.2,
    #                      do_ns=True,run_type=3,pop_tournment=2)
    
    #acts =player.mcts_act(env,1000,20,do_ns=True)
    #acts =player.mo_mcts_act(env,1000,20)
    #acts =player.pareto_mcts_act(env,500,20)

run()


