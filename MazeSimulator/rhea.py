import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import random
import math
import logging
from random import randint
import copy
import ns
import pygame

import ea
from operator import itemgetter
import time
from maze_simulator import MazeSimulator

do_debug=False


class RollingHorizonEvolutionaryAlgorithm():

    def __init__(self, rollout_actions_length, environment, mutation_probability, num_pop, use_shift_buffer=False,
                 flip_at_least_one=True, discount_factor=1, ignore_frames=0,do_ns = False,path=''):

        self._logger = logging.getLogger('RHEA')

        self._rollout_actions_length = rollout_actions_length
        self._environment = MazeSimulator(render=True, xml_file=path)
        self.path = path
        self._environment.env.robot = copy.deepcopy(environment.env.robot)             
        self.environment = environment
        self._use_shift_buffer = use_shift_buffer
        self._flip_at_least_one = flip_at_least_one
        self._mutation_probability = mutation_probability
        self._discount_factor = discount_factor
        self.num_pop = num_pop
        self._ignore_frames = ignore_frames
        self.best_solution_val = -99999
        self.best_solution = None
        self.cur_best_solution_val = -99999
        self.cur_best_solution = None
        self.cur_best_novelty = -99999
        self.cur_best_novelty_sol = None

        self.history = []
        self.ns = ns.NoveltySearch()
        self.old_pop=[]
        self.ns = ns.NoveltySearch(behavior_type='ad_hoc')
        #self.ns = ns.NoveltySearch(behavior_type='trajectory')
        #self.ns = ns.NoveltySearch(behavior_type='hamming')
        #self.ns = ns.NoveltySearch(behavior_type='entropy')
        #self.ns = ns.NoveltySearch(behavior_switch=True)
        self.ea = ea.EA('default')
        self.do_ns = do_ns
        #self.behv_state =copy.deepcopy(environment)
        #self.behv_last_visit =copy.deepcopy(environment)
        #self.behv_rewards =copy.deepcopy(environment)
        self.playout_count=0

        # Initialize the solution to a random sequence
        if self._use_shift_buffer:
            self._solution = self._random_solution()

        self.tree = {}
        self.store_tree = False

    def evaluate_rollouts(self,env,solutions,discount=1,ignore_frames = 0,):
        rewards = []
        nv_rews= []
        behvs=[]
        backup_env = MazeSimulator(render=False, xml_file=self.path)
        backup_env.env.robot = copy.deepcopy(env.env.robot) 
        
        for sol in solutions:
            return_r = 0
            return_n = 0
            acts = []

            for act in sol:
                acts.append(act)
                #time.sleep(0.005)
                env.render()
                actions=[a for a in env.get_possible_actions()]
                _,_,done = env.step(actions[act-1],0.2)
                pygame.event.pump()
                r = env.evaluate_fitness()
                if(r>0):
                    print('reward: ',r)
                    a=2/0
                #return_r+=r*discount
                return_r=r

                if(self.do_ns):
                    behv = (int(env.env.robot.location[0]),int(env.env.robot.location[1]) )  
                    
                    #Vai guardando o behavior de cada passo. Quando done = True(ep acabar),
                    #adiciona todo o behavior do ep no arquivo.
                    #Mas so adiciona no arquivo se store_behavior for True
                    self.ns.build_behavior(behv,act,done,False)

                if(done):
                    break


            if(self.do_ns):
                #behv =(env.posx,env.posy)
                behvs.append(self.ns.episode_behavior)
                #return_n =self.ns.get_novelty(behv)
                #print('solution behavior: ',self.ns.episode_behavior)
                return_n=self.ns.get_approx_novelty(self.ns.episode_behavior,k=1000,done=True)
                
 
                #nv_rews.append(return_n)
                nv_rews.append(return_r*0 + return_n*1.0)

                print('novelty reward: ',return_n)
                print('distance reward: ',return_r)

                self.playout_count+=1
                
                if(done==False):
                    self.ns.build_behavior(behv,act,True,False)

            #Store in tree if allowed
            if(self.store_tree):
                self.expand_tree(sol,return_r)
            #############Evaluating Rollouts by ##########
            #############rewards and/or Diversity ########

            #caso novelty, a rec(novelty do behv) e so no fim
            # do rollout

            #salvando melhor rollout e melhor retorno obtido
            #de todos os tempos
            if(return_r>self.best_solution_val):
                if(do_debug):
                    print('best reward now: ',return_r)
                #self.best_solution = np.concatenate((np.asarray(self.history),sol))
                self.best_solution = self.history +acts
                self.best_solution_val = return_r

            #salvando sol mais diversa e com maior nivel de
            #diversidade. Deve ser usado na pop atual apenas
            if(return_n>self.cur_best_novelty):
                #print('best novelty from current pop: ',return_n)
                self.cur_best_novelty_sol = self.history +acts
                self.cur_best_novelty = return_n

            #salvando melhor retorno e sol da pop atual
            if(return_r>self.cur_best_solution_val):
                #self.best_solution = np.concatenate((np.asarray(self.history),sol))
                self.cur_best_solution = self.history +acts
                self.cur_best_solution_val = return_r

            rewards.append(return_r)

            env.env.robot =copy.deepcopy(backup_env.env.robot)

        
        if(self.do_ns):
            return np.asarray(nv_rews),behvs

        return np.asarray(rewards),behvs

    def pop_evolution(self,pop):
        new_pop = []
        
        #rhea
        if(len(pop)>1):

            if(len(self.old_pop)==0):
                self.old_pop = pop

            pop_scores=[]
            #pega metade da pop(a melhor metade, pois esta ordenada)
            print('best half: ',len(pop)//2)
            for i in range(len(pop)//2):
                #couple = [pop[i],pop[i+1]]
                couple = [pop[random.randint(0,len(pop)-1)],pop[random.randint(0,len(pop)-1)]]
                #print('couple',couple)
                offsprings = self.ea.crossover(couple,cross_type='uniform')
                #offsprings=couple
                offsprings = self.ea.mutateNN(self._environment, offsprings,self._mutation_probability)
                for j in range(len(offsprings)):
                    new_pop.append(offsprings[j])
            print('new pop: ',len(new_pop))

            #avalia a nova populacao
            new_pop_behvs=[]
            for i in range(len(new_pop)):
                score,behvs = self.evaluate_rollouts(self._environment,[new_pop[i]], self._discount_factor, self._ignore_frames)
                pop_scores.append([new_pop[i],score,behvs])
                new_pop_behvs.append(behvs)

            #avalia e coloca a velha populacao
            for i in range(len(self.old_pop)):
                score,behvs = self.evaluate_rollouts(self._environment,[self.old_pop[i]], self._discount_factor, self._ignore_frames)
                pop_scores.append([self.old_pop[i],score,behvs])
            

            #print('pop_scores: ',pop_scores)
            pop_scores=sorted(pop_scores,key=itemgetter(1))
            pop_scores.reverse()

            print('sorted pop_scores: ',len(pop_scores))

            print('Stored new ',len(new_pop_behvs),' behaviors')
            
            #guarda os behaviors das novas solucoes somente
            for i in range(len(new_pop_behvs)):
                if(self.ns.behavior_switch):
                    self.ns.set_behavior_in_archive(new_pop_behvs[i],self.ns.behavior_archives[self.ns.index],True)
                else:
                    self.ns.set_behavior_in_archive(new_pop_behvs[i],self.ns.behavior_archive,True)

            #pega metade dos melhores(da soma da pop nova+pop velha) 

            self.old_pop=[]#limpa velha populacao
            new_pop=[]
            for i in range(len(pop_scores)//2):
                self.old_pop.append(pop_scores[i][0])#nova velha pop
                new_pop.append(pop_scores[i][0])
            print('final new pop: ',len(new_pop))
            """
        #random hill mutation climber
        if(len(pop)==1):
            
            
            offspring = self.ea.mutate(pop[0].tolist(),False,self._environment,self._mutation_probability)
            mutated_score,behvs = self.evaluate_rollouts(self._environment,[offspring], self._discount_factor, self._ignore_frames)
            if(do_debug):
                print('best novelty sol evaluation:')
            
            if(self.ns.behavior_switch):
                if(len(self.cur_best_novelty_solutions[self.ns.behavior_type])==0):
                    self.cur_best_novelty_solutions[self.ns.behavior_type] =offspring

            if(len(self.cur_best_novelty_sol)==0):
                self.cur_best_novelty_sol =offspring 

            if(self.ns.behavior_switch):
                best_novelty,_ = self.evaluate_rollouts(self._environment,
                                                        [self.cur_best_novelty_solutions[self.ns.behavior_type]],
                                                        self._discount_factor, self._ignore_frames,
                                                        store_behavior=False)
            else:
                best_novelty,_ = self.evaluate_rollouts(self._environment,[self.cur_best_novelty_sol],
                                                        self._discount_factor, self._ignore_frames,
                                                        store_behavior=False)
            print('mutated_score: ',mutated_score,', best_novelty: ',best_novelty)
            if(mutated_score>best_novelty):
                print('new best mutation score: ',mutated_score)
                self.cur_best_novelty_sol=offspring
                new_pop=[np.asarray(offspring)]
            else:
                #print('cur best novelty score updated: ',best_novelty[0])
                new_pop=[np.asarray(self.cur_best_novelty_sol)]
        """
        return new_pop


    def _random_solution(self):
        """
        Create a random set fo actions
        """
        possible_actions =[0,1,2,3]
        #possible_actions =self._environment.get_possible_actions())
        l = len(possible_actions)
        return np.array([possible_actions[randint(0,l-1)] for _ in range(self._rollout_actions_length)])

    def expand_tree(self,individual,value):
        history = ""
        for i in range(len(individual)):
            history+=str(individual[i])+','
            if(history in self.tree):
                self.tree[history]+=value
            else:
                self.tree[history] = value

    def get_best_tree_action(self,env):
        #possible_actions =env.get_possible_actions()
        possible_actions = [0,1,2,3]
        values = []
        for i in range(len(possible_actions)):
            values.append( self.tree[str(possible_actions[i])+','] )
        print(values)
        best_idx = np.argmax(np.asarray(values), axis=0)
        print("Best Tree value: ",values[best_idx])

        return possible_actions[best_idx]


    #mutation hill climber
    def run3(self,it=1,pop_evolution=5,use_shift=False,store_tree = False):
        print('run3')
        pop =[]
        self.store_tree = store_tree
        for k in range(it):
            print(k)
            self.tree={}
            actions = []
            score = 0

            if(use_shift):
                if(k==0):
                    for i in range(self.num_pop):
                        pop.append(self._random_solution())
                else:
                    #shiftbuffer
                    print('rollout len before:'+str(len(pop[0])))
                    pop[0]= self._shift_and_append(pop[0])
                    print('rollout len after:'+str(len(pop[0])))

            else:
                pop =[]
                for i in range(self.num_pop):
                    pop.append(self._random_solution())


            for i in range(pop_evolution):
                if(i%10==0):
                    #print(self.ns.behavior_archive)
                    print('evolution: ',i)
                    print("Archive size: ",len(self.ns.behavior_archive))
                    

                #self.process(pre_sol =solution)
                pop=self.pop_evolution(pop)

            
            #print('##############')
            #print(self.best_solution_val)
            #print('pop size: ',len(pop))
            #print(pop[0])
            if(store_tree):
                act = self.get_best_tree_action(self.environment)
                self.environment.step(act)
            else:
                #print(pop[0][0])
                #a=2/0
                actions=[a for a in self.environment.get_possible_actions()]
                print(actions)
                c=0
                for p in pop[0]:
                    c+=1
                    #if(c>10):
                    #    break
                    #print(pop[0][0])
                    self.environment.step(actions[p-1],0.2)
                #self.environment.render()
                #pygame.event.pump()

            #self._environment = copy.deepcopy(self.environment)
            self._environment.env.robot = copy.deepcopy(self.environment.env.robot)             
        
            #print(self.environment.render())
