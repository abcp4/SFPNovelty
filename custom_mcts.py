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
import selection
import multiobj
import update
import expand

def debug(behv_state,behv_last_visit,behv_rewards,r,p,state):
    behv_state.maze[state.posx,state.posy]+=1
    behv_last_visit.maze[state.posx,state.posy]=p
    behv_rewards.maze[state.posx,state.posy]=r

class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.nv_value = 0
        self.amaf_value = 0
        self.amaf_nv_value = 0
        self.pareto_front = [0,0]
        self.amaf_pareto_front = [0,0]
        self.amaf_visits =0 
        self.hv = 0
        

class Runner:
    def __init__(self,  max_depth=1000, playouts=10000,do_ns=False,sel_type = 'ucb',update_type ='amaf'):

        self.max_depth = max_depth
        self.playouts = playouts
        #self.ns = ns.NoveltySearch(behavior_type='ad_hoc')
        self.ns = ns.NoveltySearch(behavior_type='trajectory')
        #self.ns = ns.NoveltySearch(behavior_type='hamming')
        #self.ns = ns.NoveltySearch(behavior_type='entropy')
        #self.ns = ns.NoveltySearch(behavior_switch=True)
        self.do_ns =do_ns
        self.cycle =0
        self.e=1
        self.decay=0.99995
        self.sel_type = sel_type
        self.update_type =update_type

    def run(self,env):

        best_rewards = []
        start_time = time()

        root = Node(None, None)



        best_actions = []
        best_reward = float("-inf")
        sel = selection.Selection()
        behv_state =copy.deepcopy(env)
        behv_last_visit =copy.deepcopy(env)
        behv_rewards =copy.deepcopy(env)
        for p in range(self.playouts):
            if(p%50==0):
                if(self.ns.behavior_switch):
                    print('exchanging behavior')
                    self.ns.switch_behavior()

                print(p)
            state = copy.deepcopy(env)
            sum_reward = 0
            amaf_actions ={}
            node = root
            terminal = False
            actions = []

            sel_size=0


            # selection
            while node.children:
                #sel_size+=1
                #if sel_size > self.max_depth:
                #    break

                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = sel.select(node.children,ucb_type=self.sel_type,e = self.e)

                    self.e*=self.decay
               
                _, reward, terminal = state.step(node.action)
                if(self.do_ns):
                    behv = (state.posx,state.posy)
                    self.ns.build_behavior(behv,node.action,False,False)
                    #self.ns.put_behavior(behv,action)
                sum_reward += reward
                actions.append(node.action)
                amaf_actions[str(node.action)]=1

            # expansion
            if not terminal:
                node.children =expand.default(node,state)
                #node.children = [Node(node, a) for a in state.get_possible_actions()]
                random.shuffle(node.children)

            # playout
            while not terminal:
                pactions =state.get_possible_actions()
                action =pactions[random.randint(0,len(pactions)-1)]
    
                _, reward, terminal  = state.step(action)
                if(self.do_ns):
                    behv = (state.posx,state.posy)
                    #self.ns.put_behavior(behv,action)
                    self.ns.build_behavior(behv,action,False,False)
                sum_reward += reward
                actions.append(action)
                amaf_actions[str(node.action)]=1

                if len(actions) > self.max_depth:
                    sum_reward -= 100
                    break

           # remember best
            if best_reward < sum_reward:
                best_reward = sum_reward
                best_actions = actions

            #Behavior
            nv_reward = 0
            br = behv_rewards.maze[state.posx,state.posy]
            if(self.do_ns):
                behv =(state.posx,state.posy)
                #nv_reward =self.ns.get_approx_novelty(behv)
                #self.ns.put_behavior(behv)
                #print('len rollout: ',len(actions))

                nv_reward =self.ns.get_approx_novelty(self.ns.episode_behavior,done=True)
                self.ns.build_behavior(behv,action,True,True)
                


                #self.ns.put_behavior(self.ns.episode_behavior,action,done=True)
                
                
                debug(behv_state,behv_last_visit,behv_rewards,nv_reward,p,state)
            else:
                debug(behv_state,behv_last_visit,behv_rewards,sum_reward,p,state)

 
            #Update
            update.backpropagate(node,sum_reward,nv_reward,amaf_actions, update_type=self.update_type)
            

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
