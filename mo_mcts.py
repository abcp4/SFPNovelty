#!/usr/bin/env python2
import os
import sys
from hv3 import HyperVolume
import random
import itertools
from time import time
import copy
from math import sqrt, log
import ns
#from pygmo import hypervolume
c =1 

def ucb(node):
    global c
    return (node.value / node.visits) + c*sqrt(log(node.parent.visits)/node.visits)

def avg(node):
    return node.value / node.visits


def dominance(P,r):
    if(len(P)==0):
        #print("First time visiting node")
        return True,[]
    #print("checking dominance:")
    #print("pareto front:", P)
    #print("Curr solution:",r)
    dominated_sols = []
    count = 0

    cond_addR =False  
    #Se tiver algum sol menor que r em algum quesito, adicionar r
    #Se uma sol for menor que r em todos os quesitos, remover a sol
    for sol in P:
        cond_removeSol =False 
        #Se a nova sol for pior que qualquer
        #sol pareto, nao adiciona-la e sair
        #if(sol[0]>r[0] and sol[1]>r[1]): 
        #    return False,[]
        
        if(sol[0]<r[0] or sol[1]<r[1]):#Se a nova sol domina em algum criterio 
            cond_addR= True
        if((sol[0]<=r[0] and sol[1]<r[1]) or (sol[0]<r[0] and sol[1]<=r[1])):
            dominated_sols.append(count)
        if((sol[0]==r[0] and sol[1]>r[1]) or (sol[0]>r[0] and sol[1]==r[1]) ):
            return False,[]
        count+=1

    return cond_addR,dominated_sols

def update(node,rs,dominated):
    #node.visits += 1

    #node.pareto_front[0] +=rs[0] 
    #node.pareto_front[1] +=rs[1] 

    #if( dominated == False):
    cond_addR,dominated_sols = dominance(node.P,rs)
    #if(not cond_addR):
    #    dominated = True
    #else:
    #print("***********Dominated ones:*************")
    for index in sorted(dominated_sols, reverse=True):
        #print(node.P[index])
        del node.P[index]
        if(len(node.P)==0 and not cond_addR):
            a =2/0

    #print("***************************************")
    #Se nova sol r nao foi dominada, adicionar ao pareto front
    if(cond_addR):
        #print("New Pareto Sol: ",rs)
        node.visits += 1
        node.pareto_front[0] +=rs[0] 
        node.pareto_front[1] +=rs[1] 
        node.P.append(rs)
        referencePoint = [0,0]
        hyperVolume = HyperVolume(referencePoint)
        front =node.P 
        result = hyperVolume.compute(front)
        #hv = hypervolume(node.P)
        #ref_point = hv.refpoint(offset = 0.1)
        #result= hv.compute(ref_point)
        node.value+=result 
    else:
        return node
    if(node.parent is not None):
        update(node.parent,rs,dominated)
    #else:
        #print("!!!!!!!!!!!!Root updated!! New Pareto Front:!!!!!!!!!!")
        #print(node.P)
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return node


class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.nv_value = 0
        self.pareto_front = [0,0]
        self.P = []

class Runner:
    def __init__(self,  max_depth=1000, playouts=10000,do_ns=False):

        self.max_depth = max_depth
        self.playouts = playouts
        self.ns = ns.NoveltySearch()
        self.do_ns =do_ns
        self.cycle =0

    def print_stats(self, loop, score, avg_time):
        sys.stdout.write('\r%3d   score:%10.3f   avg_time:%4.1f s' % (loop, score, avg_time))
        sys.stdout.flush()

    def run(self,env):
        best_rewards = []
        start_time = time()

        root = Node(None, None)

        best_actions = []
        best_reward = float("-inf")
        behv_state =copy.deepcopy(env)
        behv_last_visit =copy.deepcopy(env)
        behv_rewards =copy.deepcopy(env)
        for p in range(self.playouts):
            state = copy.deepcopy(env)
            #del state._monitor

            sum_reward = 0
            node = root
            terminal = False
            actions = []

            # selection
            while node.children:
                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = max(node.children, key=avg)
                #print(node.action)
                _, reward, terminal = state.step(node.action)
                #sum_reward += reward
                actions.append(node.action)

            # expansion
            if not terminal:
                #node.children = [Node(node, a) for a in combinations(state.action_space)]
                node.children = [Node(node, a) for a in state.get_possible_actions()]
                random.shuffle(node.children)

            # playout
            while not terminal:
                pactions =state.get_possible_actions()
                action =pactions[random.randint(0,len(pactions)-1)]
    
                _, reward, terminal  = state.step(action)
                #sum_reward += reward
                actions.append(action)

                if len(actions) > self.max_depth:
                    sum_reward =reward
                    break

           # remember best
            if best_reward < sum_reward:
                best_reward = sum_reward
                best_actions = actions
            
            behv_state.maze[state.posx,state.posy]+=1
            behv_last_visit.maze[state.posx,state.posy]=p
            #behavior
            behv =(state.posx,state.posy)
            nv_reward =self.ns.get_novelty(behv)
            self.ns.put_behavior(behv)
            br = behv_rewards.maze[state.posx,state.posy]
            behv_rewards.maze[state.posx,state.posy]=nv_reward

            
            # backpropagate
            #print('obj1: game reward:',sum_reward)
            #print('obj2: novelty reward:',nv_reward)
            node = update(node,[sum_reward,nv_reward],False)

            sum_reward = 0
            nv_reward = 0

        print('Final Pareto Front')
        print(root.P)
        print('##############')
        print(behv_state.render())
        print('last visit')
        
        print(behv_last_visit.render())
        print('last reward')
        print(behv_rewards.render())
        return best_actions
        '''
        for action in best_actions:
            _, reward, terminal  = env.step(action)
            sum_reward += reward
            if terminal:
                break

        best_rewards.append(sum_reward)
        score = max(moving_average(best_rewards, 100))
        avg_time = (time()-start_time)/(loop+1)
        self.print_stats(loop+1, score, avg_time)'''


