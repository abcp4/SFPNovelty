#!/usr/bin/env python2
import os
import sys
from hv3 import HyperVolume
import random
from random import shuffle
import itertools
from time import time
import copy
from math import sqrt, log
import ns

c = 1


def dominated(r,P):
    dom =True
    for sol in P:
        if(sol[0]<r[0] or sol[1]<r[1]): 
            dom= False

        if((sol[0]==r[0] and sol[1]==r[1]) or (sol[0]==r[0] and sol[1]==r[1]) ):
            dom=False
        #se r for dominado, sair
        if((sol[0]==r[0] and sol[1]>r[1]) or (sol[0]>r[0] and sol[1]==r[1]) ):
            return True 

    return dom 


def pareto_ucb(nodes,e):
    pareto_front = []
    node_front = []
    n = 0
    for node in nodes:
        n+=node.visits
    for node in nodes:
        rv = []
        for val in node.pareto_front:
            #rv.append(val / node.visits + sqrt( (4*log(n)+log(2)) / 2*node.visits))
            rv.append(val / node.visits)
        node_front.append(rv)
    
    for i in range(len(nodes)):
        sol = node_front[i]
        if(not dominated(sol,node_front)):
            pareto_front.append(i)
    #print('node front:',node_front)
    #print('pareto front:',pareto_front)
    r = random.random()
    if(r<e):
        max_novelty = 0
        index_novelty = 0
        for i in range(len(nodes)):
            sol = node_front[i]
            if(sol[1]>max_novelty):
                max_novelty = sol[1]
                index_novelty = i
        return nodes[index_novelty]

    else:
        max_value = 0
        index_value = 0
        #for i in range(len(nodes)):
        #    sol = node_front[i]
            #if(sol[0]>max_value):
            #    max_value = sol[0]
            #    index_value = i
        #return nodes[index_value]
        return max(nodes, key=ucb)

def ucb(node):
    return node.value / node.visits + 1*sqrt(log(node.parent.visits)/node.visits)


def avg(node):
    return node.nv_value / node.visits


def update(node,rs,dominated):
    node.visits += 1
    node.pareto_front[0] +=rs[0] 
    node.pareto_front[1] +=rs[1] 
    node.value+=rs[0]
    if(node.parent is not None):
        update(node.parent,rs,dominated)
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
        self.e = 1
        self.decay =0.99995


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
                    self.e = self.e*self.decay
                    node = pareto_ucb(node.children,self.e)
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
                sum_reward += reward
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

        print('e:',self.e)
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


