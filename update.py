
from hv3 import HyperVolume

from multiobj import dominance 


def backpropagate(node,sum_reward,nv_reward,visited_nodes,update_type = 'default'):
    if(update_type == 'default'):
        return update_normal(node,sum_reward,nv_reward)
    if(update_type == 'momcts'):
        return mo_update(node,[sum_reward,nv_reward])

    if(update_type == 'amaf'):
        return amaf_update(node,visited_nodes,sum_reward,nv_reward)



def update_normal(node,sum_reward,nv_reward):
    while node:
        node.pareto_front[0] += sum_reward
        node.pareto_front[1] += nv_reward
        node.visits += 1
        node.value += sum_reward
        node.nv_value += nv_reward
        node =node.parent

def amaf_update(node,list_moves,sum_reward,nv_reward):
    while node:
        node.pareto_front[0] += sum_reward
        node.pareto_front[1] += nv_reward
        node.visits += 1
        node.value += sum_reward
        node.nv_value += nv_reward
        for child in node.children:
            if str(child.action) in list_moves:
                child.pareto_front[0] += sum_reward
                child.pareto_front[1] += nv_reward
                child.visits += 1
                child.value += sum_reward
                child.nv_value += nv_reward
        
        node =node.parent

def rave_update(node,list_moves,sum_reward,nv_reward):
    while node:
        node.pareto_front[0] += sum_reward
        node.pareto_front[1] += nv_reward
        node.visits += 1
        node.value += sum_reward
        node.nv_value += nv_reward
        for child in node.children:
            if str(child.action) in list_moves:
                child.amaf_pareto_front[0] += sum_reward
                child.amaf_pareto_front[1] += nv_reward
                child.amaf_visits += 1
                child.amaf_value += sum_reward
                child.amaf_nv_value += nv_reward
        

        node =node.parent


def mo_update(node,rs):
   
    cond_addR,dominated_sols = dominance(node.P,rs)
    #if(not cond_addR):
    #    dominated = True
    #else:
    #print("***********Dominated ones:*************")
    for index in sorted(dominated_sols, reverse=True):
        #print(node.P[index])
        del node.P[index]
        if(len(node.P)==0 and not cond_addR):
            a =2/0

    #print("***************************************")
    #Se nova sol r nao foi dominada, adicionar ao pareto front
    if(cond_addR):
        #print("New Pareto Sol: ",rs)
        node.visits += 1
        node.pareto_front[0] +=rs[0] 
        node.pareto_front[1] +=rs[1] 
        node.P.append(rs)
        referencePoint = [0,0]
        hyperVolume = HyperVolume(referencePoint)
        front =node.P 
        result = hyperVolume.compute(front)
        #hv = hypervolume(node.P)
        #ref_point = hv.refpoint(offset = 0.1)
        #result= hv.compute(ref_point)
        node.hv+=result 
    else:
        return node
    if(node.parent is not None):
        update(node.parent,r)
    #else:
        #print("!!!!!!!!!!!!Root updated!! New Pareto Front:!!!!!!!!!!")
        #print(node.P)
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return node



