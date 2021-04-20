
import numpy as np
import random 
import copy 
from random import randint

do_debug=False
class EA():
    def __init__(self,etype):
        self.etype = etype


    def mutate(self,solution,flip_at_least_one,environment, mutation_probability):
        """
        Mutate one solution into n
        """
        # Create a set of indexes in the solution that we are going to mutate
        mutation_indexes = set()
        if(do_debug):
            print('sol:',solution)
        solution_length = len(solution)
        if flip_at_least_one:
            mutation_indexes.add(np.random.randint(solution_length))

        mutation_indexes = mutation_indexes.union(
            set(np.where(np.random.random([solution_length]) < mutation_probability)[0]))

        # Create the number of mutations that is the same as the number of mutation indexes
        num_mutations = len(mutation_indexes)
        p_actions=[0,1,2,3]
        mutations = [p_actions[randint(0,len(p_actions)-1)] for _ in range(num_mutations)]

        # Replace values in the solutions with mutated values
        new_solution = np.copy(solution)
        new_solution[list(mutation_indexes)] = mutations

        return new_solution


    def mutateN(self,num_rollout_pop,solution,flip_at_least_one,environment,mutation_probability):
        """
        Mutate one solution into n
        """
        #print('before mutation: ',solution)
        candidate_solutions = []
        # Solution here is 2D of rollout_actions x batch_size
        for b in range(num_rollout_pop):
            # Create a set of indexes in the solution that we are going to mutate
            mutation_indexes = set()
            solution_length = len(solution)
            if flip_at_least_one:
                mutation_indexes.add(np.random.randint(solution_length))

            mutation_indexes = mutation_indexes.union(
                set(np.where(np.random.random([solution_length]) < mutation_probability)[0]))

            # Create the number of mutations that is the same as the number of mutation indexes
            num_mutations = len(mutation_indexes)
            #l = len(environment.get_possible_actions())
            #possiveis acoes. Varia de cada env
            p_actions=[0,1,2,3]
            #print('l: ',l)
            mutations = [p_actions[randint(0,len(p_actions)-1)] for _ in range(num_mutations)]

            # Replace values in the solutions with mutated values
            new_solution = np.copy(solution)
            new_solution[list(mutation_indexes)] = mutations
            #print('after mutation: ',new_solution)
            candidate_solutions.append(new_solution)

        return np.stack(candidate_solutions)

    def mutateNN(self,environment,solutions,mutation_probability,flip_at_least_one=True):
        """
        Mutate N solutions into N
        """

        # Solution here is 2D of rollout_actions x batch_size
        #print()
        for solution in solutions:
            # Create a set of indexes in the solution that we are going to mutate
            mutation_indexes = set()
            solution_length = len(solution)
            if flip_at_least_one:
                mutation_indexes.add(np.random.randint(solution_length))

            mutation_indexes = mutation_indexes.union(
                set(np.where(np.random.random([solution_length]) < mutation_probability)[0]))

            # Create the number of mutations that is the same as the number of mutation indexes
            num_mutations = len(mutation_indexes)
            l = len(environment.get_possible_actions())
            #Importante!! alterar para cada jogo
            possible_actions=[0,1,2,3]
            l =len(possible_actions)
            #l = len(environment.get_possible_actions())
            #possible_actions =environment.get_possible_actions()
            mutations = [possible_actions[randint(0,l-1)] for _ in range(num_mutations)]

            # Replace values in the solutions with mutated values
            if(do_debug):
                print('solution before mutation',solution)
            solution[list(mutation_indexes)] = mutations
            if(do_debug):
                print('solution after mutation',solution)
        return np.stack(solutions)


    def crossover (self,couple, cross_type='point'):
        off1=[]
        off2=[]

        if (cross_type == 'point'):
            #int p = gen.nextInt(actions.length - 3) + 1;
            p = random.randint(0,len(couple[0]))
            #p = len(couple[0])//2
            for i in range(len(couple[0])):
                if (i < p):
                    off1.append(couple[0][i])
                    off2.append(couple[1][i])
                else:
                    off1.append(couple[1][i])
                    off2.append(couple[0][i])
            
        elif (cross_type == 'uniform'):
            #escolhe acoes aleatorias de um dos dois   
            #if p1==0:
            #    p2=1
            #else:
            #    p2=0
            for i in range(len(couple[0])):
                p = random.randint(0,1)
                if(p==0):
                    off1.append(couple[0][i])
                    off2.append(couple[1][i])
                else:
                    off1.append(couple[1][i])
                    off2.append(couple[0][i])
        else:
            return np.asarray(couple)
        return np.asarray([off1,off2])



    

