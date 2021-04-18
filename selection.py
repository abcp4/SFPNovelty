import random
import math
from multiobj import dominated

class Selection:

    def select(self,nodes,ucb_type='ucb',e = 1):


        if(ucb_type=='pareto'):
            return pareto_ucb(nodes,e)
        if(ucb_type=='moucb'):
            return max(nodes, key=moucb)
        if(ucb_type=='ucb2'):
            return ucb2(nodes)
        if(ucb_type=='ucb'):
            return max(nodes, key=ucb)
        if(ucb_type=='rave'):
            return rave_sel(nodes)
        if(ucb_type=='rave_novelty'):
            return rave_sel(nodes,novelty=True)
        if(ucb_type=='avg_reward'):
            return max(nodes, key=avg)
        if(ucb_type=='avg_novelty'):
            return max(nodes, key=avgn)
        if(ucb_type == 'egreedy'):
            r =random.random()
            if(r<e):
                node = max(nodes, key=avgn)
            else:
                node = max(nodes, key=ucb)
            return node


def pareto_ucb(nodes,e):
    pareto_front = []
    node_front = []
    n = 0
    for node in nodes:
        n+=node.visits
    for node in nodes:
        rv = []
        for val in node.pareto_front:
            #rv.append(val / node.visits + sqrt( (4*log(n)+log(2)) / 2*node.visits))
            rv.append(val / node.visits)
        node_front.append(rv)
    
    for i in range(len(nodes)):
        sol = node_front[i]
        if(not dominated(sol,node_front)):
            pareto_front.append(i)
    #print('node front:',node_front)
    #print('pareto front:',pareto_front)
    r = random.random()
    if(r<e):
        max_novelty = 0
        index_novelty = 0
        for i in range(len(nodes)):
            sol = node_front[i]
            if(sol[1]>max_novelty):
                max_novelty = sol[1]
                index_novelty = i
        return nodes[index_novelty]

    else:
        max_value = 0
        index_value = 0
        #for i in range(len(nodes)):
        #    sol = node_front[i]
            #if(sol[0]>max_value):
            #    max_value = sol[0]
            #    index_value = i
        #return nodes[index_value]
        return max(nodes, key=ucb)

def ucb(node):
    return node.value / node.visits + 1*math.sqrt(math.log(node.parent.visits)/node.visits)

def moucb(node):
    return node.hv / node.visits + 2*math.sqrt(math.log(node.parent.visits)/node.visits)


def avg(node):
    return node.value / node.visits


def avgn(node):
    return node.nv_value / node.visits

#https://github.com/pbsinclair42/MCTS/blob/master/mcts.py
def ucb2(nodes, explorationConstant=1/ math.sqrt(2)):
    bestValue = float("-inf")
    bestNodes = []
    for child in nodes:
        nodeValue = child.value / child.visits + explorationConstant * math.sqrt(
            2 * math.log(child.parent.visits) / child.visits)
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)

def rave_sel(nodes,novelty=False):
    bestValue = float("-inf")
    bestNodes = []
    k=100
    for child in nodes:
        v =child.amaf_visits
        if(novelty):
            val = child.nv_value
            amaf_val = child.amaf_nv_value
        else:
            val = child.value
            amaf_val = child.amaf_value
        if(v==0):
            v =1
        b = math.sqrt(k/(3*(child.parent.visits)+k))
        nodeValue =(1-b)*(val/child.visits)+b*(amaf_val/v)
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)
