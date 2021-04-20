import numpy as np
import sys
import time
import random
import copy


def rollout(env,steps = 100):
    ret = 0
    for t in range(steps):
        actions =env.get_possible_actions()
        act =actions[random.randint(0,len(actions)-1)]
        obs,r,done = env.step(act)
        ret+=r
        if done:
            break
    return ret


def rollout_act(env):
    actions =env.get_possible_actions()
    bestval = -999999
    bestact = actions[0]
    values =[-999999, -999999, -999999,-999999] 
    copy_env = copy.deepcopy(env)
    for i in range(len(actions)):
        val = 0
        for j in range(5):
            #first step
            obs,r, done = copy_env.step(actions[i])
            new_val = rollout(copy_env,10)
            val+=new_val
            copy_env = copy.deepcopy(env)
        values[actions[i]] = val
        if(val>bestval):
            bestact = actions[i]
            bestval = val
    print('values: ',values)
    return bestact

def uct_act(env,it,rollout_limit,discount=0.9,e=1):
    from mcp import UCT
    uct = UCT(it,rollout_limit,discount,e)
    act =uct.get_act('',env)
    return act
def mcts_act(env,playouts,depth,do_ns=False):
    from mcts import Runner
    #return Runner(playouts=playouts, max_depth=depth,do_ns=do_ns).run(env)
    return Runner(playouts=playouts, max_depth=depth,do_ns=do_ns)

def mo_mcts_act(env,playouts,depth,do_ns=False):
    from mo_mcts import Runner
    return Runner(playouts=playouts, max_depth=depth).run(env)

def pareto_mcts_act(env,playouts,depth):
    from pareto_mcts import Runner
    return Runner(playouts=playouts, max_depth=depth).run(env)

def rhea_act(env,it=10,pop_evolution=5,pop_num=10,rollout_limit=20,mutation_prob = 0.1,use_shift_buffer=False,do_ns=False,run_type = 0,path='hardmaze_env.xml'):
    from rhea import RollingHorizonEvolutionaryAlgorithm
    re = RollingHorizonEvolutionaryAlgorithm(rollout_limit,env,mutation_prob,
                                            pop_num,use_shift_buffer=use_shift_buffer,
                                            do_ns=do_ns,path=path)
    
    if(run_type==1):
        return re.run()
    if(run_type==2):
        return re.run2(it=it)
    if(run_type==3):
        return re.run3(it=it,pop_evolution=pop_evolution)