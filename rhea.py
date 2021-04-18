import numpy as np
import random
import math
import logging
from random import randint
import copy
import ns
import ea
from operator import itemgetter



"""
O Grande problema do rhea com NS e que a evolucao da populacao
depende do fitness da solucao, que geralmente e estavel.
O novelty nao e estavel no entanto, ja que quanto mais uma solucao
e visitada, menor e sua novelty. Elitismo(que procura manter 
solucoes boa entre geracoes de populacoes) nao e efetivo pelo mesmo
motivo, ja que nao da pra garantir que a solucao mantida entre
geracoes vai ser aquela com real potencial de gerar mais novelty.
Uma sol e tentar utilizar hall of the fame
"""

do_debug=False

def debug(behv_state,behv_last_visit,behv_rewards,r,p,state):
    behv_state.maze[state.posx,state.posy]+=1
    behv_last_visit.maze[state.posx,state.posy]=p
    behv_rewards.maze[state.posx,state.posy]=r

class RollingHorizonEvolutionaryAlgorithm():

    def __init__(self, rollout_actions_length, environment, mutation_probability, num_pop, use_shift_buffer=False,
                 flip_at_least_one=True, discount_factor=1, ignore_frames=0,do_ns = False,pop_tournment=3):

        self._logger = logging.getLogger('RHEA')

        self._rollout_actions_length = rollout_actions_length
        self._environment = copy.deepcopy(environment)
        self.environment = copy.deepcopy(environment)
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
        self.cur_best_novelty_sol = []
        self.cur_best_novelty_solutions = {'ad_hoc':[],'trajectory':[],'hamming':[],'entropy':[]}


        self.history = []
        self.old_pop=[]
        self.ns = ns.NoveltySearch(behavior_type='ad_hoc')
        #self.ns = ns.NoveltySearch(behavior_type='trajectory')
        #self.ns = ns.NoveltySearch(behavior_type='hamming')
        #self.ns = ns.NoveltySearch(behavior_type='entropy')
        #self.ns = ns.NoveltySearch(behavior_switch=True)
        self.ea = ea.EA('default')
        self.do_ns = do_ns
        self.behv_state =copy.deepcopy(environment)
        self.behv_last_visit =copy.deepcopy(environment)
        self.behv_rewards =copy.deepcopy(environment)
        self.playout_count=0

        # Initialize the solution to a random sequence
        if self._use_shift_buffer:
            self._solution = self._random_solution()

        self.tree = {}
        self.store_tree = False
        self.pop_tournment = pop_tournment
        
        #TODO: hall of fame
        self.hof = []
        self.max_rew=-9999

    def evaluate_rollouts(self,env,solutions,discount=1,ignore_frames = 0,store_behavior=True):
        rewards = []
        nv_rews= []
        behvs=[]

        for sol in solutions:
            run_env = copy.deepcopy(env)
            return_r = 0
            acts = []

            for act in sol:
                acts.append(act)
                _,r,done = run_env.step(act)
                #return_r+=r*discount
                
                if(self.do_ns):
                    behv = (run_env.posx,run_env.posy)
                    
                    #Vai guardando o behavior de cada passo. Quando done = True(ep acabar),
                    #adiciona todo o behavior do ep no arquivo.
                    #Mas so adiciona no arquivo se store_behavior for True
                    self.ns.build_behavior(behv,act,done,store_behavior)
                
                return_r+=r
                if(done):
                    break

            if(self.do_ns):
                #armazena o behavior completo do episodio
                if(self.ns.behavior_switch):
                    behvs.append(self.ns.episode_behaviors[self.ns.index])
                else:
                    #print('appending: ',self.ns.episode_behavior)
                    behvs.append(self.ns.episode_behavior)
                #return_r=self.ns.get_novelty_simple(self.ns.episode_behavior,done=True)
                
                #caso o ep n tenha terminado com done=True, forcar o done
                if(done==False):  
                    #if(self.ns.behavior_type == 'trajectory'):
                    return_r=self.ns.get_approx_novelty(self.ns.episode_behavior,done=True)
                    self.ns.build_behavior(self.ns.episode_behavior,act,done=True,store_behavior=store_behavior)

                #reset
                self.ns.episode_behavior = []
                self.ns.episode_probs = np.zeros(30)
                #store reward
                nv_rews.append(return_r)

                self.playout_count+=1
                debug(self.behv_state,self.behv_last_visit,self.behv_rewards,return_r,self.playout_count,run_env)
                if(do_debug):
                    print('novelty: ',return_r)
                    print('behv: ',behv)
            else:
                debug(self.behv_state,self.behv_last_visit,self.behv_rewards,return_r,self.playout_count,run_env)


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
            #if(return_r>self.cur_best_novelty):
                #print('best novelty from current pop: ',return_r)
            #    self.cur_best_novelty_sol = self.history +acts
            #    self.cur_best_novelty = return_r

            #salvando melhor retorno e sol da pop atual
            if(return_r>self.cur_best_solution_val):
                #self.best_solution = np.concatenate((np.asarray(self.history),sol))
                #self.cur_best_solution = self.history +acts
                self.cur_best_solution =sol
                self.cur_best_solution_val = return_r
            rewards.append(return_r)

        if(self.do_ns):
            #print('nv_rews:',nv_rews)
            nv_rews=np.array(nv_rews)
            
            #return nv_rews,behvs
            return nv_rews,behvs

        return np.asarray(rewards),behvs


    def tournment(self,pop,k=1,p=0.7):
        sample_pop=[]
        indices =np.random.choice(len(pop),k)
        for i in range(k):
            sample_pop.append(pop[indices[i]])
        sample_scores,behvs = self.evaluate_rollouts(self._environment,sample_pop, self._discount_factor, self._ignore_frames)
        ind_scores =[]
        for i in range(k):
            ind_scores.append([i,sample_scores[i]])
        ind_scores=sorted(ind_scores,key=itemgetter(1))
        ind_scores.reverse()
        #da probs pros melhores individuos serem sel em ordem
        for i in range(k):
            x =random.uniform(0,1)
            if(x<p):
                return sample_pop[ind_scores[i][0]],[behvs[i]],np.array([sample_scores[i]])
        return sample_pop[ind_scores[0][0]],[behvs[i]],np.array([sample_scores[i]])

    def select_basic(self,pop):
        new_pop = []
        l_pop=len(pop)
        pop_scores=[]
        new_pop_behvs=[]
        for i in range(int(l_pop*(5/5))):
            #couple = [pop[i],pop[i+1]]
            couple = [pop[i]]
            #couple = [pop[random.randint(0,len(pop)-1)],pop[random.randint(0,len(pop)-1)]]
            #offsprings = self.ea.crossover(couple,cross_type='point')
            offsprings=couple
            #print(offsprings)
            offsprings = self.ea.mutateNN(self._environment, offsprings,self._mutation_probability)
            #print(offsprings)
            for j in range(len(offsprings)):
                new_pop.append(offsprings[j])
        
        for i in range(len(new_pop)):
            score,behvs = self.evaluate_rollouts(self._environment,[new_pop[i]], self._discount_factor, self._ignore_frames,store_behavior=False)
            pop_scores.append([new_pop[i],score,behvs])
            new_pop_behvs.append(behvs)

        #print('pop_scores: ',pop_scores)
        pop_scores=sorted(pop_scores,key=itemgetter(1))
        pop_scores.reverse()
        
        return pop_scores,new_pop_behvs

    def select_tournment(self,pop):
        l_pop=len(pop)
        pop_scores=[]
        new_pop_behvs=[]
        for i in range(l_pop):
            winner,behv,score = self.tournment(pop,3)
            pop_scores.append([winner,score,behv])
            new_pop_behvs.append(behv)
        pop_scores=sorted(pop_scores,key=itemgetter(1))
        pop_scores.reverse()
        return pop_scores,new_pop_behvs




    def pop_evolution(self,pop):
        
        
        #rhea
        if(len(pop)>1):            
            l_pop=len(pop)
            
            """
            ##############################################
            Selection and Evaluation
            ##############################################
            """
            pop_scores,new_pop_behvs = self.select_basic(pop)
            #pop_scores,new_pop_behvs = self.select_tournment(pop)
            
            #elitism
            elit={}
            c=0
            #colocar melhores da pop antiga
            #salvar dict dos melhores
            #for i in range(int(l_pop*(1/5))):
            #    new_pop.append(pop[c])
            #    elit[str(pop[c])]=1

            
            #print(len(pop_scores))

            #print('sorted pop_scores: ',pop_scores)
            
            """
            ##############################################
            Novelty recording
            ##############################################
            """
            c=0
            #print("*********************************")
            for i in range(len(new_pop_behvs)):
                if(str(pop_scores[i][0]) in elit):#Nao armazenar elites
                    #print('elit ',i,' score: ',pop_scores[i][1])
                    continue
                c+=1

                if(self.ns.behavior_switch):
                    self.ns.set_behavior_in_archive(new_pop_behvs[i],self.ns.behavior_archives[self.ns.index],True)
                    #self.ns.set_behavior_in_archive(new_pop_behvs[i][1],self.ns.behavior_archives[1],True)
                    
                else:
                    self.ns.set_behavior_in_archive(new_pop_behvs[i],self.ns.behavior_archive,True)

            new_pop=[]
            for i in range(len(pop_scores)):
                new_pop.append(pop_scores[i][0]) 
            #print('new pop:',len(new_pop))
            
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
            #print('new pop',new_pop)
            #print(behvs.keys())
            #self.set_behaviors(behvs,mutated_score)

        return new_pop

    def _shift_and_append(self, solution):
        """
        Remove the first element and add a random action on the end
        """
        #self.history.append(solution[0])
        #print('a:',solution)
        new_solution = solution[1:]
        l = len(self._environment.get_possible_actions())
        p_a =[0,1,2,3]
        random_act = p_a[randint(0,len(p_a)-1)]
        new_solution= np.append(new_solution,random_act)
        #print('b:',new_solution)
        return np.asarray(new_solution)

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
        #print(values)
        best_idx = np.argmax(np.asarray(values), axis=0)
        #print("Best Tree value: ",values[best_idx])

        return possible_actions[best_idx]

    #mutation hill climber
    def run3(self,it=1,pop_evolutions=10,use_shift=False,store_tree = False):
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
                    #print('rollout len before:'+str(len(pop[0])))
                    pop[0]= self._shift_and_append(pop[0])
                    #print('rollout len after:'+str(len(pop[0])))

            else:
                pop =[]
                for i in range(self.num_pop):
                    pop.append(self._random_solution())
                self.old_pop = pop

            for i in range(pop_evolutions):
                if(i%10==0):
                    #print(self.ns.behavior_archive)
                    print('evolution: ',i)
                    print("Archive size: ",len(self.ns.behavior_archive))
                    if(self.ns.behavior_switch):
                        print('exchanging behavior')
                        self.ns.switch_behavior()

                #self.process(pre_sol =solution)
                pop=self.pop_evolution(pop)

            #print('hall of fame: ',self.hof)

            #print('##############')
            #print(self.best_solution_val)
            #print('pop size: ',len(pop))
            #print(pop[0])
            if(store_tree):
                act = self.get_best_tree_action(self.environment)
                self.environment.step(act)
            else:
                self.environment.step(pop[0][0])

            self._environment = copy.deepcopy(self.environment)
            #print(self.environment.render())
        print('##############')

        print('accumulative end positions:')
        print(self.behv_state.render())

        print('last visited places')
        print(self.behv_last_visit.render())

        print('last rewards gained')
        print(self.behv_rewards.render())

        return self.cur_best_solution

