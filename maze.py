import numpy as np
import sys
import time
import random
import copy
import math
class Maze:
    def __init__(self,game_type=0):
        self.reset()
        self.game_type=game_type


    def reset(self):
        self.maze = np.asarray([[2,0,0,0,0,0,0,0,0],
                                [0,1,1,1,1,1,1,1,0],
                                [0,1,0,0,0,0,0,0,0],
                                [0,1,0,0,0,0,0,1,0],
                                [0,1,0,1,1,1,1,1,0],
                                [0,1,0,0,0,0,0,1,0],
                                [0,1,1,1,0,0,0,1,0],
                                [0,0,0,1,3,0,0,1,0]]
                                )
        
        self.maze_old = np.asarray([[2,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,0,0],
                                [0,0,0,0,0,0,0,3,0]]
                                )
        self.posx = 0
        self.posy = 0
        self.objx = 7
        self.objy = 4

        #self.objx = 7
        #self.objy = 7
        
        return self.maze
   

    def step(self,act):
        #left
        if(act == 0):
            if(self.posy!=0):
                if(self.maze[self.posx,self.posy-1]!=1):
                    self.maze[self.posx,self.posy-1] = 2
                    self.maze[self.posx,self.posy] = 0
                    self.posy-=1
        #right
        elif(act ==1):
            if(self.posy!=8):
                if(self.maze[self.posx,self.posy+1]!=1):
                    self.maze[self.posx,self.posy+1] = 2
                    self.maze[self.posx,self.posy] = 0
                    self.posy+=1
        #up
        elif(act == 2):
            if(self.posx!=0):
                if(self.maze[self.posx-1,self.posy]!=1):
                    self.maze[self.posx-1,self.posy] = 2
                    self.maze[self.posx,self.posy] = 0
                    self.posx-=1
        #down
        elif(act == 3):
            if(self.posx!=7):
                if(self.maze[self.posx+1,self.posy]!=1):
                    self.maze[self.posx+1,self.posy] = 2
                    self.maze[self.posx,self.posy] = 0
                    self.posx+=1

        reward = 0
        if(self.game_type==1):
            reward =math.sqrt(math.pow(self.posx-self.objx,2) + math.pow(self.posy-self.objy,2))
            reward =-reward

        done = False
        if(self.posx == self.objx and self.posy == self.objy):
            print('Found prize!!!')
            reward = 1000
            #self.posx =0
            #self.posy =0
            done = True

        return self.maze,reward,done

    def get_possible_actions(self):
        actions = []
        if(self.posy!=0):
            if(self.maze[self.posx,self.posy-1]!=1):
                actions.append(0)
        if(self.posy!=8):
            if(self.maze[self.posx,self.posy+1]!=1):
                actions.append(1)
        if(self.posx!=0):
            if(self.maze[self.posx-1,self.posy]!=1):
                actions.append(2)
        if(self.posx!=7):
            if(self.maze[self.posx+1,self.posy]!=1):
                actions.append(3)
          
        return actions

    def render(self):
        print(self.maze,end='\r',flush = True)
        sys.stdout.flush()
        print('')

