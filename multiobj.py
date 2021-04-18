
#Pareto MCTS
def dominated(r,P):
    dom =True
    for sol in P:
        if(sol[0]<r[0] or sol[1]<r[1]): 
            dom= False

        if((sol[0]==r[0] and sol[1]==r[1]) or (sol[0]==r[0] and sol[1]==r[1]) ):
            dom=False
        #se r for dominado, sair
        if((sol[0]==r[0] and sol[1]>r[1]) or (sol[0]>r[0] and sol[1]==r[1]) ):
            return True 

    return dom 


#MO MCTS
def dominance(P,r):
    if(len(P)==0):
        #print("First time visiting node")
        return True,[]
    #print("checking dominance:")
    #print("pareto front:", P)
    #print("Curr solution:",r)
    dominated_sols = []
    count = 0

    cond_addR =False  
    #Se tiver algum sol menor que r em algum quesito, adicionar r
    #Se uma sol for menor que r em todos os quesitos, remover a sol
    for sol in P:
        cond_removeSol =False 
        #Se a nova sol for pior que qualquer
        #sol pareto, nao adiciona-la e sair
        #if(sol[0]>r[0] and sol[1]>r[1]): 
        #    return False,[]
        
        if(sol[0]<r[0] or sol[1]<r[1]):#Se a nova sol domina em algum criterio 
            cond_addR= True
        if((sol[0]<=r[0] and sol[1]<r[1]) or (sol[0]<r[0] and sol[1]<=r[1])):
            dominated_sols.append(count)
        if((sol[0]==r[0] and sol[1]>r[1]) or (sol[0]>r[0] and sol[1]==r[1]) ):
            return False,[]
        count+=1

    return cond_addR,dominated_sols

