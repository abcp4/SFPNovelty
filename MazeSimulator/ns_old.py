
import math
import random

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


    def get_novelty_simple(self,behv):
        novelty = 0
        num = 0
        for ba in self.behavior_archive:
            d = self.get_distance(ba,behv)
            novelty+=d
            num+=1
        if(num==0):
            return 0
        return novelty/num

    def get_approx_novelty(self,behv,k=100):
        novelty = 0
        num = 0
        distances =[]
        #print(len(self.behavior_archive))
        for ba in self.behavior_archive:
            d = self.get_distance(ba,behv)
            distances.append(d)
            num+=1
        if(num==0):
            return 0
        distances.sort()
        #print('distances from', behv[0],behv[1],':',distances)
        sumk=0

        if(len(distances)<k):
            k =len(distances)
        for i in range(k):
            sumk+=distances[i]
        return sumk/k

