#!/usr/bin/env python2
import os
import sys
import random
import itertools
from time import time
import copy
from math import sqrt, log
import ns
import random

def ucb(node):
    return node.value / node.visits + 10*sqrt(log(node.parent.visits)/node.visits)

def avg(node):
    return node.value / node.visits

#Ser o segundo termo se trata da exploracao, e ele vai ser alto com acoes
#poucos exploradas. No entanto, estamos buscando por inovacao, e escolher
#as acoes que nao inovam, equivalem a não explorar. É uma contradicao,
#logo ucb ou qualquer mov q explora pode falhar quando o objetivo é
#a  busca por inovacao(novelty). Devemos exploitar novelty.

#def ucbn(node):
#    return node.nv_value / node.visits + 1.4*sqrt(log(node.parent.visits)/node.visits)

def avgn(node):
    return node.nv_value / node.visits

class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.nv_value = 0


class Runner:
    def __init__(self,  max_depth=1000, playouts=10000,do_ns=False):

        self.max_depth = max_depth
        self.playouts = playouts
        self.ns = ns.NoveltySearch()
        self.do_ns =do_ns
        self.cycle =0
        self.e=1
        self.decay=0.99995

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
            print("Playout: ",p)
            state = copy.deepcopy(env)
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
                    if(not self.do_ns):
                        #node = max(node.children, key=avg)
                        node = max(node.children, key=ucb)
                    else:
                        #node = max(node.children, key=avgn)
                        r =random.random()
                        if(r<self.e):
                        #   print('here')
                            node = max(node.children, key=avgn)
                        else:
                            node = max(node.children, key=ucb)
                        self.e*=self.decay
               
                #print(node.action)
                _, reward, terminal = state.step(node.action)
                sum_reward += reward
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
                    #sum_reward -= 100
                    break

           # remember best
            if best_reward < sum_reward:
                best_reward = sum_reward
                best_actions = actions
            
            behv_state.maze[state.posx,state.posy]+=1
            behv_last_visit.maze[state.posx,state.posy]=p
            nv_reward = 0
            #behavior
            self.cycle+=1
            #if(self.cycle ==200):
            #    self.do_ns=not self.do_ns
            #    self.cycle=0

            br = behv_rewards.maze[state.posx,state.posy]
            if(self.do_ns):
                behv =(state.posx,state.posy)
                #nv_reward =self.ns.get_novelty(behv)
                nv_reward =self.ns.get_novelty_simple(behv)
                self.ns.put_behavior(behv)
                behv_rewards.maze[state.posx,state.posy]=nv_reward
            else:
                behv_rewards.maze[state.posx,state.posy]=sum_reward

            
            # backpropagate
            #print('reward:',sum_reward)
            while node:
                node.visits += 1
                node.value += sum_reward
                node.nv_value += nv_reward
                node =node.parent
            

        sum_reward = 0
        nv_reward = 0
        print('e: ',self.e)
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

    
