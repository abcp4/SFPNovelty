import numpy as np
import sys
import time
import random
import copy
import maze
import player
env = maze.Maze(1)
env.render()    
def steps():
    while(1):
        #act = random.randint(0,3)
        #act = player.rollout_act(env)
        #act =player.mcts_act(env,100,10,do_ns=True)[0]
        act =player.mo_mcts_act(env,100,10)[0]
        #act = player.uct_act(env,100,15,discount=0.9,e=0.1)
        print('possible actions:',env.get_possible_actions())
        _,r,_=env.step(act)
        print('reward:',r)
        env.render()
        time.sleep(1)
def run():
    acts =player.mcts_act(env,10000,100,do_ns=False)
    #acts =player.mcts_act(env,500,20,do_ns=True)
    #acts =player.mo_mcts_act(env,1000,20)
    #acts =player.pareto_mcts_act(env,500,20)
    for act in acts:
        _,r,_=env.step(act)
        print('reward:',r)
        env.render()
        time.sleep(1)
run()


