#!/usr/bin/env python2
import os
import sys
import random
import itertools

import copy
from math import sqrt, log
import ns
import random
from maze_simulator import MazeSimulator
import pygame
import time


#https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
def getBestChild(node, explorationConstant=1/ sqrt(2)):
    bestValue = float("-inf")
    bestNodes = []
    for child in node.children:
        nodeValue = child.value / child.visits + explorationConstant * sqrt(
            2 * log(node.visits) / child.visits)
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)

def ucb(node):
    return node.value / node.visits + 2*sqrt(log(node.parent.visits)/node.visits)
    #return node.value / node.visits + sqrt(2*log(node.parent.visits)/node.visits)


def avg(node):
    return node.value / node.visits


def avgn(node):
    return node.nv_value / node.visits


def half(node):
    return (node.value / node.visits)*0.2 +(node.nv_value / node.visits)*0.8

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
        self.ns = ns.NoveltySearch(behavior_type='ad_hoc')
        #self.ns = ns.NoveltySearch(behavior_type='trajectory')
        #self.ns = ns.NoveltySearch(behavior_type='hamming')
        #self.ns = ns.NoveltySearch(behavior_type='entropy')
        #self.ns = ns.NoveltySearch(behavior_switch=True)
        self.do_ns =do_ns
        self.cycle =0
        self.e=1
        self.decay=1


    def run(self,env,path):
        best_rewards = []
        start_time = time.time()

        root = Node(None, None)

        best_actions = []
        best_reward = float("-inf")
        state = MazeSimulator(render=True, xml_file=path)
        real_move=50
        c=0
        for p in range(self.playouts):
            print("Playout: ",p)
            state.env.robot = copy.deepcopy(env.env.robot) 
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
                        #node = max(node.children, key=ucb)
                        node = getBestChild(node)
                    else:
                        r =random.random()
                        if(r<self.e):
                            #node = max(node.children, key=avgn)
                            node = max(node.children, key=half)
                        else:
                            
                            node = getBestChild(node)
                            #node = max(node.children, key=ucb)
                        self.e*=self.decay
               
                #print(node.action)
                #time.sleep(0.005)
                state.render()
                _, _, terminal = state.step(node.action,0.2)

                if(self.do_ns):
                    behv = (int(state.env.robot.location[0]),int(state.env.robot.location[1]) ) 
                    self.ns.build_behavior(behv,node.action,False,False)

                pygame.event.pump()
                reward = state.evaluate_fitness()
                if(reward>0):
                    print('reward: ',reward)
                    a=2/0
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
    
                _, _, terminal = state.step(action,0.2)
                reward = state.evaluate_fitness()
                #sum_reward += reward
                actions.append(action)

                if(self.do_ns):
                    behv = state.env.robot.location
                    self.ns.build_behavior(behv,action,False,False)

                if len(actions) > self.max_depth:
                    sum_reward -= 100
                    break
            sum_reward = state.evaluate_fitness()

            # remember best
            #if best_reward < sum_reward:
            #    best_reward = sum_reward
            #    best_actions = actions
            
            nv_reward = 0
            #behavior
            if(self.do_ns):
                behv = (int(state.env.robot.location[0]),int(state.env.robot.location[1]) )  
                    
                #behv =(state.posx,state.posy)
                #nv_reward =self.ns.get_novelty(behv)

                self.ns.build_behavior(behv,action,False,store_behavior=False)
                nv_reward=self.ns.get_approx_novelty(self.ns.episode_behavior,done=True)
                self.ns.build_behavior(behv,action,True,store_behavior=True)
                #self.ns.set_behavior_in_archive(behv,self.ns.behavior_archive,True)
                
                
                #self.ns.put_behavior(behv)
                
            
            # backpropagate
            print('reward:',sum_reward)
            print('nv_reward:',nv_reward)
            while node:
                node.visits += 1
                node.value += sum_reward
                node.nv_value += nv_reward
                node =node.parent
            

        sum_reward = 0
        nv_reward = 0
        print('e: ',self.e)
        best_actions = actions
        
        return best_actions
        