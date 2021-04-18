import copy
import numpy as np
import random

class UCT():
    def __init__(self,it=10,rollout_limit=50,discount=0.9,e=1):
        self.rollout_limit=rollout_limit
        self.it = it
        self.state_space = {}
        self.discount = discount
        self.e =e

    def add_node(self,state,act):
        if(str(state)+str(act) not in self.state_space):
            #see if it will return the reference since it is a tuple(deep list will)
            self.state_space[str(state)+str(act)] = [0,0]
        return self.state_space[str(state)+str(act)]

    def update(self,node,val):
        node[0]+=1
        node[1]+=val

    def get_val(self,state,act):
        if(str(state)+str(act) not in self.state_space):
            return -999#evita acoes nao disp de serem selecionadas
        node=self.state_space[str(state)+str(act)]

        #return  node[0]/node[1]
        return  node[1]

    def get_best_act(self,env,state,e=0,debug=False):
        actions = env.get_possible_actions()
        #actions = [0,1,2]
        if(random.randint(0,1000)<1000*e):
            return actions[random.randint(0,len(actions)-1)]
       
        vals = [-9999,-9999,-9999,-9999]
        bestval = -9999
        bestact = 0
        for i in range(len(actions)):
            vals[actions[i]] = self.get_val(state,actions[i])
        for i in range(len(vals)):
            if(vals[i]>bestval):
                bestact = i
                bestval = vals[i]
        if(debug):
            print('vals:',vals)

        #total greedy
        return bestact
       

    def rollout(self,state,env):
        trajectory = []
        ret = 0
        for i in range(0,self.rollout_limit):
            act = self.get_best_act(env,state,1)
            _,r,done = env.step(act)
            #state is chain of actions
            state+=str(act)
            ret+=r
            node = self.add_node(state,'')
            trajectory.append(node)
            if(done):
                break
        return trajectory,ret


    def get_act(self,state,env):
        copy_env = copy.deepcopy(env)
        init_state = copy.deepcopy(state)
        for i in range(self.it):
            copy_env = copy.deepcopy(env)
            trajectory,ret = self.rollout(state,copy_env)
            for node in trajectory:
                ret=ret*self.discount
                self.update(node,ret)
        act = self.get_best_act(env,init_state,0,True)
        return act
