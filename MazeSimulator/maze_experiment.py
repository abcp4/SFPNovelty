import pygame
import time
from maze_simulator import MazeSimulator
import random
import copy
import math
import player


class NoveltySearch():
    def __init__(self):
        self.behavior_archive = []
        self.max_size = 1000
        self.cur_size = 0
    
    def put_behavior(self,behv):
        if(self.cur_size >= self.max_size):
            self.behavior_archive = self.behavior_archive[1:]    
        self.behavior_archive.append(behv)
        self.cur_size+=1
    def pop_behavior(self):
        return self.behavior_archive.pop()
    def reset_archive(self):
        self.behavior_archive = []


    def get_distance(self,b1,b2):
        return math.sqrt( (b2[0] - b1[0])**2 + (b2[1] - b1[1])**2 )


    def get_novelty(self,behv):
        novelty = 0
        num = 0
        for ba in self.behavior_archive:
            d = self.get_distance(ba,behv)
            novelty+=d
            num+=1
        if(num==0):
            return 0
        return novelty/num

def get_behavior(sim):
    return sim.env.robot.location


class UCT():
    def __init__(self,limit=50):
        self.limit = limit
        self.state_space = {}

    def add_node(self,state,act):
        if(self.state_space[str(state)+str(act)] is None):
            #see if it will return the reference since it is a tuple(deep list will)
            self.state_space[str(state)+str(act)] = (0,0)
        return self.state_space[str(state)+str(act)]

    def update(self,node,val):
        node[0]+=1
        node[1]+=val/node[0]



    def get_possible_actions(self,state):
        return [0,1,2,3]


    def get_best_act(self,state):
        #actions = get_possible_actions(state)
        actions = [0,1,2]
        if(random.randint(0,1000)>950):
            return actions[random.randint(0,2)]
        
        vals = [0,1,2]
        bestval = -9999
        bestact = 0
        for i in range(len(actions)):
            vals[i] = self.state_space[str(state)+str(actions[i])]
            if(vals[i]>bestval):
                bestact = actions[i]
                bestval = vals[i]

        #total greedy
        return bestact
        

    def rollout(self,state,env):
        trajectory = []
        ret = 0
        for i in range(0,self.limit):
            act = get_best_act(state)
            state,r = env.step(act, 0.2)
            ret+=r
            node = self.add_node(state,act)
            trajectory.append(node)
        return trajectory,ret


    def get_act(self,state,env):
        copy_env = copy.deepcopy(env)
        init_state = copy.deepcopy(state)
        trajectory,ret = rollout(state,copy_env)
        for node in trajectory:
            self.update(node,ret)
        act = get_best_act(init_state)
        return act



def main():
    sim = MazeSimulator(render=True, xml_file='hardmaze_env.xml')

    for t in range(30000):
        time.sleep(0.005)
        sim.render()
        keys = pygame.key.get_pressed()
        action = [0, 0.4, 0]
        if keys[pygame.K_LEFT]:
            action = [0.05, 0.2, 0]
        if keys[pygame.K_RIGHT]:
            action = [0, 0.2, 0.05]

        finder_obs, radar_obs, done = sim.step(action, 0.2)

        pygame.event.pump()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(sim.evaluate_fitness())
            break


def main2():
    sim = MazeSimulator(render=False, xml_file='hardmaze_env.xml')
    
    for t in range(1000):
        print(t)
        #time.sleep(0.005)
        #sim.render()
        act = random.randint(0,2)
        if(act==0):
            action = [0, 0.4, 0]
        elif(act==1):
            action = [0.05, 0.2, 0]
        elif(act==2):
            action = [0, 0.2, 0.05]

        finder_obs, radar_obs, done = sim.step(action, 0.2)

        #pygame.event.pump()
        if done:
            break
    print("Episode finished after {} timesteps".format(t+1))
    print(sim.evaluate_fitness())

def rollout(sim,steps = 100):
    for t in range(steps):
        act = random.randint(0,2)
        if(act==0):
            action = [0, 0.4, 0]
        elif(act==1):
            action = [0.05, 0.2, 0]
        elif(act==2):
            action = [0, 0.2, 0.05]

        finder_obs, radar_obs, done = sim.step(action, 0.2)
        if done:
            break
    return sim.evaluate_fitness(),get_behavior(sim)



#objective based
def main3():
    sim = MazeSimulator(render=True, xml_file='hardmaze_env.xml')
    csim = MazeSimulator(render=False, xml_file='hardmaze_env.xml')

    for t in range(10000):
        print(t)
        actions = [[0, 0.4, 0],[0.05, 0.2, 0],[0, 0.2, 0.05]]
        time.sleep(0.005)
        sim.render()
        bestval = -9999
        bestact = actions[0]
        values = [0,0,0]
        for i in range(len(actions)):
            val = 0
            for j in range(5):
                #first step
                finder_obs, radar_obs, done = csim.step(actions[i], 0.2)
                new_val,behv = rollout(csim,20)
                val+=new_val
                csim.env.robot = copy.deepcopy(sim.env.robot) 
            values[i] = val
            if(val>bestval):
                bestact = actions[i]
                bestval = val
        print('values: ',values)
            
        sim.step(bestact, 0.2)
        print('Reward: ',sim.evaluate_fitness())

        pygame.event.pump()
        if done:
            break
    print("Episode finished after {} timesteps".format(t+1))
    print(sim.evaluate_fitness())
            
def main4():
    sim = MazeSimulator(render=True, xml_file='hardmaze_env.xml')
    csim = MazeSimulator(render=False, xml_file='hardmaze_env.xml')
    bs = NoveltySearch()
    backup_robot = copy.deepcopy(sim.env.robot)

    for t in range(10000):
        #if(t%500==0):
        #    sim.env.robot = copy.deepcopy(backup_robot)

        print(t)
        print('archive size: ',len(bs.behavior_archive))
        #bs.reset_archive()
        actions = [[0, 0.4, 0],[0.05, 0.2, 0],[0, 0.2, 0.05]]
        time.sleep(0.005)
        sim.render()
        bestval = -9999
        bestact = actions[0]
        values = [0,0,0]
        actions_behvs = [[],[],[]]
        for i in range(len(actions)):
            val = 0

            for j in range(2):
                #first step
                finder_obs, radar_obs, done = csim.step(actions[i], 0.2)
                new_val,behv = rollout(csim,10)

                #comment block below for fitness only
                #new_val = bs.get_novelty(behv) 
                bs.put_behavior(behv)
                actions_behvs[i].append(behv)
                
                csim.env.robot = copy.deepcopy(sim.env.robot) 
        best_val = 0
        for i in range(len(actions_behvs)):
            for behv in actions_behvs[i]:
                v = bs.get_novelty(behv)
                values[i]+=v

                #if(v>best_val):
                #    bestact = actions[i]
                #    best_val = v

            if(values[i]>bestval):
                bestval = values[i]
                bestact = actions[i]
        print('values:',values)
        if(sum(values)<1):
            sim.env.robot = copy.deepcopy(backup_robot)


            
        sim.step(bestact, 0.2)

        pygame.event.pump()
        if done:
            break
    print("Episode finished after {} timesteps".format(t+1))
    print(sim.evaluate_fitness())

def main5():
    sim = MazeSimulator(render=True, xml_file='hardmaze_env.xml')
    csim = MazeSimulator(render=False, xml_file='hardmaze_env.xml')
    
    for t in range(30000):
        time.sleep(0.005)
        sim.render()
        keys = pygame.key.get_pressed()
        action = [0, 0.4, 0]
        if keys[pygame.K_LEFT]:
            action = [0.05, 0.2, 0]
        if keys[pygame.K_RIGHT]:
            action = [0, 0.2, 0.05]

        actions = [[0, 0.4, 0],[0.05, 0.2, 0],[0, 0.2, 0.05]]
        values = [0,0,0]
        for i in range(len(actions)):
            finder_obs, radar_obs, done = csim.step(actions[i], 0.2)
            val = 0
            for j in range(20):
                new_val,behv = rollout(sim,5)
                val+=new_val
                csim.env.robot = copy.deepcopy(sim.env.robot) 
            values[i] = val
        print('values: ',values)

        finder_obs, radar_obs, done = sim.step(action, 0.2)

        pygame.event.pump()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print(sim.evaluate_fitness())
            break


#MCTS Novelty or UCT
def main6(path):
    #sim = MazeSimulator(render=False, xml_file=path)
    csim = MazeSimulator(render=False, xml_file=path)
    #m = player.mcts_act(csim,500,20,do_ns=True)
    #actions = m.run(csim,path)
    
    for i in range(1):
        m =player.mcts_act(csim,1000,20,do_ns=True)
        actions = m.run(csim,path)
        actions=actions
        for act in actions:
            #print('lol')
            #time.sleep(0.005)
            #sim.render()
            _,_,done = csim.step(act, 0.2)
            #print('Reward: ',sim.evaluate_fitness())

            pygame.event.pump()
    
    #print("Episode finished after {} timesteps".format(t+1))
    #print(sim.evaluate_fitness())

#RHEA Novelty
def main7(path):
    #sim = MazeSimulator(render=False, xml_file=path)
    csim = MazeSimulator(render=False, xml_file=path)
    player.rhea_act(csim,it=1,pop_evolution=300,pop_num=6,rollout_limit=40,mutation_prob = 0.1,
                          do_ns=True,run_type=3,path=path)
if __name__ == "__main__":
    #main6('hardmaze_env.xml')
    #main7('hardmaze_env.xml')
    #main7('Maze_Xml/maze_0.xml')    
    #main6('Maze_Xml/maze_9.xml')    
    main7('Maze_Xml/maze_9.xml')    