
import math
import random
import numpy as np
import editdistance
import scipy
from scipy.spatial import distance
from scipy.stats import entropy
from operator import itemgetter

def levenshtein_distance(v1,v2):
    x1=str(v1).replace(',','').replace(' ','')
    x2=str(v2).replace(',','').replace(' ','')
    return editdistance.eval(x1,x2)

def hamming_distance(v1,v2):
    #print('v1: ',len(v1[0]))
    #print('v2: ',len(v2))
    if(len(v2)<len(v1)):
        v2=np.concatenate((v2,np.zeros(len(v1)-len(v2))),axis=0)
    if(len(v2)>len(v1)):
        v1=np.concatenate((v1,np.zeros(len(v2)-len(v1))),axis=0)
    return distance.hamming(v1,v2)

def euclidian_distance(v1,v2):
    return np.linalg.norm(np.asarray(v1)-np.asarray(v2))


class NoveltySearch():
    def __init__(self,min_score=-1,behavior_type='ad_hoc',behavior_switch=False):
        self.behavior_archive = []
        self.behavior_cache={}
        self.max_size = 3000
        self.cur_size = 0
        self.min_score=min_score
        self.behavior_type=behavior_type
        self.behavior_switch=behavior_switch
        self.behaviors_types=['ad_hoc','trajectory','hamming','entropy']
        if(self.behavior_switch):
            self.behavior_archives = [[],[],[],[]]
            self.episode_behaviors = [[],[],[],[]]
            index=random.randint(0,3)
            self.behavior_type = self.behaviors_types[index]
            self.behavior_archive  = self.behavior_archives[index]
            self.episode_probs = np.zeros(30)

            self.best_novelty_solutions = [[],[],[],[]]
            self.best_novelty_behv_values = {'ad_hoc':0,'trajectory':0,'hamming':0,'entropy':0}


        if self.behavior_type=='hamming':
            self.episode_behavior = np.zeros(0)
        elif self.behavior_type=='entropy':
            self.episode_behavior = np.zeros(0)
            self.episode_probs = np.zeros(30)
        else:
            self.episode_behavior=[]

        self.index= 0
        self.process_distances='all'

    #o switch tem q ser triggered fora do objeto ns
    def switch_behavior(self):
        if(self.behavior_switch):
            index=random.randint(0,3)
            self.index = index
            print('Now behavior: ',index)
            self.behavior_type = self.behaviors_types[index]
            self.behavior_archive  = self.behavior_archives[index]
            self.episode_probs = np.zeros(30)


    def get_adhoc_behavior(self,behv,episode_behavior,done):
        if(done==False):
            episode_behavior=behv
        return episode_behavior

    def get_trajectory_behavior(self,behv,episode_behavior,done):
        if(done==False):
            #print(episode_behavior)
            for b in behv:
                episode_behavior.append(b)
        return episode_behavior

    def get_hamming_behavior(self,behv,episode_behavior,act,done):
        if(done==False):
            feature_vec=np.zeros(30)
            feature_vec[behv[0]]=1
            feature_vec[behv[1]+10]=1
            feature_vec[act+20]=1
            episode_behavior=np.concatenate((episode_behavior,feature_vec), axis=0)

        return episode_behavior


    def get_entropy_behavior(self,behv,episode_behavior,episode_probs,act,done):
        #episode_probs=np.zeros(30)
        if(done==False):
            episode_probs[behv[0]]+=1
            episode_probs[behv[1]+10]+=1
            episode_probs[act+20]+=1

            sum1,sum2,sum3 = 0,0,0
            for i in range(10):
                sum1+=episode_probs[i]
                sum2+=episode_probs[i+10]
                sum3+=episode_probs[i+20]

            #calcula probabilidades
            ep1 = np.copy(episode_probs[:10])/sum1
            ep2 = np.copy(episode_probs[10:20])/sum2
            ep3 = np.copy(episode_probs[20:])/sum3

            #calcula entropia
            entropy_vector=[0,0,0]
            #modificar a base para o numero de valores discretos que podem ser tomados
            #ex: acao sao 4 valores no maze 1
            entropy_vector[0]=entropy(ep1, base=10)
            entropy_vector[1]=entropy(ep2, base=10)
            entropy_vector[2]=entropy(ep3, base=10)
            episode_behavior = entropy_vector
            #print('ep probabilities: ',episode_probs)
            #print('parcial entropy vector: ',entropy_vector)
            #episode_behavior=np.concatenate((episode_behavior,entropy_vector), axis=0)


        return episode_behavior,episode_probs

    def set_behavior_in_archive(self,behv,behavior_archive,done,min_score=-1):
        if(done==True):
            if(len(behavior_archive) >= self.max_size):
                behavior_archive = behavior_archive[1:]
            behavior_archive.append(behv)
            #self.behavior_cache[str(behv)] = 1

    def build_behavior(self,behv,act=-1,done=False,store_behavior=True,behv_score=0):

        if(self.behavior_switch):
            #print(self.episode_behaviors[0])
            self.episode_behaviors[0]=self.get_adhoc_behavior(behv,self.episode_behaviors[0],done)
            if(store_behavior):
                self.set_behavior_in_archive(self.episode_behaviors[0],self.behavior_archives[0],done)
            if(done==True):
                self.episode_behaviors[0]=[]

            self.episode_behaviors[1]=self.get_trajectory_behavior(behv,self.episode_behaviors[1],done)
            if(store_behavior):
                self.set_behavior_in_archive(self.episode_behaviors[1],self.behavior_archives[1],done)
            if(done==True):
                self.episode_behaviors[1]=[]

            self.episode_behaviors[2]=self.get_hamming_behavior(behv,self.episode_behaviors[2],act,done)
            if(store_behavior):
                self.set_behavior_in_archive(self.episode_behaviors[2],self.behavior_archives[2],done)
            if(done==True):
                self.episode_behaviors[2]=np.zeros(1)

            self.episode_behaviors[3],self.episode_probs=self.get_entropy_behavior(
                                                    behv,self.episode_behaviors[3],self.episode_probs
                                                    ,act,done)
            if(store_behavior):
                self.set_behavior_in_archive(self.episode_behaviors[3],self.behavior_archives[3],done)
            if(done==True):
                self.episode_behaviors[3]=np.zeros(3)
                self.episode_probs = np.zeros(30)


        #adhoc dos ambientes de mazes so considerar a coordenada final
        elif(self.behavior_type=='ad_hoc'):
            self.episode_behavior=self.get_adhoc_behavior(behv,self.episode_behavior,done)
            if(store_behavior):
                self.set_behavior_in_archive(self.episode_behavior,self.behavior_archive,done)
            if(done==True):
                self.episode_behavior=[]

        elif(self.behavior_type=='trajectory'):
            self.episode_behavior=self.get_trajectory_behavior(behv,self.episode_behavior,done)
            if(store_behavior):
                #print('traj behv: ',self.episode_behavior)
                self.set_behavior_in_archive(self.episode_behavior,self.behavior_archive,done)

            if(done==True):
                self.episode_behavior=[]

        elif(self.behavior_type=='hamming'):
            self.episode_behavior=self.get_hamming_behavior(behv,self.episode_behavior,act,done)
            if(store_behavior):
                self.set_behavior_in_archive(self.episode_behavior,self.behavior_archive,done)

            if(done==True):
                self.episode_behavior=np.zeros(1)

        elif(self.behavior_type=='entropy'):
            self.episode_behavior,self.episode_probs=self.get_entropy_behavior(
                                                    behv,self.episode_behavior,self.episode_probs
                                                    ,act,done)
            if(store_behavior):
                self.set_behavior_in_archive(self.episode_behavior,self.behavior_archive,done)

            #if(done==True):
            #    self.episode_behavior=np.zeros(3)
            #    self.episode_probs = np.zeros(30)

    def pop_behavior(self):
        return self.behavior_archive.pop()
    def reset_archive(self):
        self.behavior_archive = []

    def get_distance(self,b1,b2):
        if(self.behavior_type=='ad_hoc'):
            if type(b1)==list:
                b1=b1[0]
            return math.sqrt( (b2[0] - b1[0])**2 + (b2[1] - b1[1])**2 )
        elif(self.behavior_type=='trajectory'):
            return levenshtein_distance(b1,b2)
        elif(self.behavior_type=='hamming'):
            return hamming_distance(b1,b2)
        elif(self.behavior_type=='entropy'):
            return euclidian_distance(b1,b2)

    #Todo score de novelty so pode ser computado no fim do episodio
    def get_novelty_simple(self,behv,done=False):
        if(done):
            novelty = 0
            num = 0
            for ba in self.behavior_archive:
                #print('b1: ',ba)
                #print('b2: ',behv)
                d = self.get_distance(ba,behv)
                #print('novelty: ',d)

                novelty+=d
                num+=1
            if(num==0):
                return 0
            return novelty/num
        else:
            return 0

    def get_approx_novelty(self,behv,k=100,done=False):
        if(self.behavior_switch):
            if self.behavior_type=='ad_hoc':
                behv = self.episode_behaviors[0]
            elif self.behavior_type=='trajectory':
                behv = self.episode_behaviors[1]
            elif self.behavior_type=='hamming':
                behv = self.episode_behaviors[2]
            elif self.behavior_type=='entropy':
                behv = self.episode_behaviors[3]

        if(done):
            novelty = 0
            num = 0
            distances =[]
            if(self.process_distances=='all'):
                for ba in self.behavior_archive:
                    if self.behavior_type=='hamming' or self.behavior_type=='entropy':
                        if(len(ba)==1):
                            ba = ba[0]

                    d = self.get_distance(ba,behv)
                    distances.append(d)
                    num+=1
            elif(self.process_distances=='sample'):
                samples=np.random.choice(len(self.behavior_archive), int(0.40*len(self.behavior_archive)),replace=False )
                for s in samples:
                    ba = self.behavior_archive[s]
                    if self.behavior_type=='hamming' or self.behavior_type=='entropy':
                        if(len(ba)==1):
                            ba = ba[0]

                    d = self.get_distance(ba,behv)
                    distances.append(d)
                    num+=1


            if(num==0):
                return 0
            distances.sort()
            #indices, distances = zip(*sorted(enumerate(distances), key=itemgetter(1)))
            vals=[]
            #for i in range(len(indices)):
            #    vals.append(self.behavior_archive[indices[i]])
            #    if(i==100):
            #        break
            #vals.append(len(self.behavior_archive))

            #print('distances: ',vals)
            #distances.reverse()
            sumk=0

            if(len(distances)<k):
                k =len(distances)
            for i in range(k):
                sumk+=distances[i]

            return sumk/k
        else:
            return 0


#d=levenshtein_distance([0,0,1],[0,0,1])
#print(d)
