
import math
import random
import numpy as np
import editdistance
import scipy
from scipy.spatial import distance
from scipy.stats import entropy

def levenshtein_distance(v1,v2):
    x1=str(v1).replace(',','').replace(' ','')
    x2=str(v2).replace(',','').replace(' ','')
    return editdistance.eval(x1,x2)

def hamming_distance(v1,v2):
    print('v1: ',v1.shape)
    print('v2: ',v2.shape)
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
        self.max_size = 1000
        self.cur_size = 0
        self.min_score=min_score
        self.behavior_type=behavior_type
        self.count_it = 0
        self.behavior_switch=behavior_switch
        self.behaviors_types=['ad_hoc','trajectory','hamming','entropy']
        if(self.behavior_switch):
            self.behavior_archives = [[],[],[],[]]
            index=random.randint(0,3)
            self.behavior_type = self.behaviors_types[index]
            self.behavior_archive  = self.behavior_archives[index]

        if self.behavior_type=='hamming':
            self.episode_behavior = np.zeros(1)
        elif self.behavior_type=='entropy':
            self.episode_behavior = np.zeros(3)
            self.episode_probs = np.zeros(30)
        else:
            self.episode_behavior=[]

    #o switch tem q ser triggered fora do objeto ns
    def switch_behavior(self):
        if(self.behavior_switch):
            index=random.randint(0,3)
            print('Now behavior: ',index)
            self.behavior_type = self.behaviors_types[index]
            self.behavior_archive  = self.behavior_archives[index]

            if self.behavior_type=='hamming':
                self.episode_behavior = np.zeros(1)
            elif self.behavior_type=='entropy':
                self.episode_behavior = np.zeros(3)
                self.episode_probs = np.zeros(30)
            else:
                self.episode_behavior=[]

            
    def put_behavior(self,behv,act=-1,done=False,behv_score=0):

        #adhoc dos ambientes de mazes so considerar a coordenada final
        if(self.behavior_type=='ad_hoc'):

            if(done==True):
                if(self.cur_size >= self.max_size):
                    self.behavior_archive = self.behavior_archive[1:]
                if self.min_score>0:
                    if(behv_score>self.min_score):
                        self.behavior_archive.append(behv)
                else:
                    self.behavior_archive.append(behv)
                self.cur_size+=1
            else:
                self.episode_behavior=behv

        elif(self.behavior_type=='trajectory'):
            #A cada interacao, coordenada como (0,1) vira [0,1].
            #Ao fim, trajetorias como [(0,1),(3,2),(1,6)] viram [0,1,3,2,1,6],
            #vira um vetor de numeros somente para facilitar no calculo de distancia
            if(done==False):
                for b in behv:
                    self.episode_behavior.append(b)

            elif(done==True):
                if(self.cur_size >= self.max_size):
                    self.behavior_archive = self.behavior_archive[1:]
                if self.min_score>0:
                    if(behv_score>self.min_score):
                        self.behavior_archive.append(self.episode_behavior)
                else:
                    self.behavior_archive.append(self.episode_behavior)
                
                self.episode_behavior=[]
                self.cur_size+=1

        elif(self.behavior_type=='hamming'):
            if(done==False):
                feature_vec=np.zeros(30)
                feature_vec[behv[0]]=1
                feature_vec[behv[1]+10]=1
                feature_vec[act+20]=1
                self.episode_behavior=np.concatenate((self.episode_behavior,feature_vec), axis=0)

            elif(done==True):
                if(self.cur_size >= self.max_size):
                    self.behavior_archive = self.behavior_archive[1:]
                if self.min_score>0:
                    if(behv_score>self.min_score):
                        self.behavior_archive.append(self.episode_behavior)
                else:
                    self.behavior_archive.append(self.episode_behavior)
                
                self.episode_behavior=np.zeros(1)
                self.cur_size+=1

        elif(self.behavior_type=='entropy'):
            if(done==False):
                self.episode_probs[behv[0]]+=1
                self.episode_probs[behv[1]+10]+=1
                self.episode_probs[act+20]+=1
                self.count_it+=1

                sum1,sum2,sum3 = 0,0,0
                for i in range(10):
                    sum1+=self.episode_probs[i]
                    sum2+=self.episode_probs[i+10]
                    sum3+=self.episode_probs[i+20]
                #calcula probabilidades
                for i in range(10):
                    self.episode_probs[i]/=sum1
                    self.episode_probs[i+10]/=sum2
                    self.episode_probs[i+20]/=sum3

                #calcula entropia
                entropy_vector=[0,0,0]
                #modificar a base para o numero de valores discretos que podem ser tomados
                #ex: acao sao 4 valores no maze 1
                entropy_vector[0]=entropy(self.episode_probs[:10], base=10)
                entropy_vector[1]=entropy(self.episode_probs[10:20], base=10)
                entropy_vector[2]=entropy(self.episode_probs[20:], base=10)
                self.episode_behavior = entropy_vector
                
                
            elif(done==True):
                if(self.cur_size >= self.max_size):
                    self.behavior_archive = self.behavior_archive[1:]
                if self.min_score>0:
                    if(behv_score>self.min_score):
                        self.behavior_archive.append(self.episode_behavior)
                else:
                    self.behavior_archive.append(self.episode_behavior)
                
                self.episode_behavior=np.zeros(3)
                self.episode_probs = np.zeros(30)
                self.cur_size+=1
    
    def pop_behavior(self):
        return self.behavior_archive.pop()
    def reset_archive(self):
        self.behavior_archive = []

    def get_distance(self,b1,b2):
        if(self.behavior_type=='ad_hoc'):
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
        if(done):
            novelty = 0
            num = 0
            distances =[]
            #print(len(self.behavior_archive))
            for ba in self.behavior_archive:
                #print('b1: ',ba)
                #print('b2: ',behv)
                d = self.get_distance(ba,behv)
                #print('novelty: ',d)
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
        else:
            return 0

#d=levenshtein_distance([0,0,1],[0,0,1])
#print(d)