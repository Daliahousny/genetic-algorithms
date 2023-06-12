import random
import numpy as np

def randomBit():  # function to decide the item is selected or not
    n = random.randint(0,1)
    if n < 0.5:
        return 0
    else:
        return 1


# generate random float number between 0 and 1

def randConstValue(a, b):  # generate constant values Pc(0.4-0.7) & rc
    n = random.uniform(a, b)
    return n

def swap(bit):
    if bit == 1:
        return 0
    else:
        return 1

def InitializePop(Num_iteration, Num_item, k, weight):
    Population= np.zeros([Num_iteration, Num_item])
    i = 0
    while (i < Num_iteration):
        sum = 0
        j = 0
        while (j < Num_item):
            Population[i][j] = randomBit()
            sum += Population[i][j] * weight[j]
            j += 1
        if sum <= k:
            i += 1

    return Population

# step 2
def calcFitness(Population, values, num_cells):
    fitness = np.zeros(num_cells)
    bestFitness = 0
    f = 0
    for chromosome in Population:
        sum = 0
        w = 0 #iterator
        for gene in chromosome:
            sum += gene * values[w]
            w += 1
        fitness[f] = sum
        if sum > bestFitness:
            bestFitness = sum
        f += 1
    return fitness


# step 3
def selection(fitness):
    cumulativeFitness = np.cumsum(fitness)
    bestCum = cumulativeFitness[-1]
    # roulette wheel
    roulette1 = random.randint(0, bestCum)
    roulette2 = random.randint(0, bestCum)
    chosen_v = 0  # variable to store the cumulative of chosen chromosome
    first_c = 0  # store index of first chosen chromosome
    second_c = 0  # store index of second chosen chromosome

    for i in range(len(cumulativeFitness)):
        if roulette1 >= chosen_v and roulette1 < cumulativeFitness[i]:
            first_c = i
            break
        chosen_v = cumulativeFitness[i]
    z = 0
    for i in range(len(cumulativeFitness)):
        if roulette2 >= chosen_v and roulette2 < cumulativeFitness[i]:
            second_c = i
            break
        chosen_v = cumulativeFitness[i]
    return first_c, second_c


def crossover(chromosomes):
    ch1 = chromosomes[0]
    ch2 = chromosomes[1]
    parent1 = Population[ch1]
    parent2 = Population[ch2]

    parent1List = parent1.tolist()
    parent2List = parent2.tolist()
    Coffspring1 = []
    Coffspring2 = []

    crossPoint = random.randint(0, len(values) - 1)
    rc = randConstValue(0.0, 1.0)
    pc = randConstValue(0.4, 0.7)

    if rc < pc:

        for i in range(crossPoint):
            Coffspring1.append(parent1List[i])

        for i in range(crossPoint, len(parent1List)):
            Coffspring1.append(parent2List[i])

        for i in range(crossPoint):
            Coffspring2.append(parent2List[i])

        for i in range(crossPoint, len(parent2List)):
            Coffspring2.append(parent1List[i])

        offspring1 = Coffspring1
        offspring2 = Coffspring2

        return offspring1, offspring2
    else:
        offspring1 = parent1
        offspring2 = parent2
    return offspring1, offspring2


def mutate1offspring(offsprings):
    newOffspring = []
    for i in offsprings:
        rm = randConstValue(0.0, 1.0)
        pm = randConstValue(0.001, 0.1)
        if rm < pm:
            newOffspring.append(swap(i))
        else:
            newOffspring.append(i)
    return newOffspring


def calcmutationFitness(newOffspring): #calculate the fitness of one offspring after mutation
    fit = 0
    for i in range(len(values)):
        fit += newOffspring[i] * weight[i]
    return fit

def mutation(offsprings, k): #mutate new offsprings and calculate their fitness
    newOffspring1 = mutate1offspring(offsprings[0])
    newOffspring2 = mutate1offspring(offsprings[1])
    fitNewOffspring1 = calcmutationFitness(newOffspring1)
    while fitNewOffspring1 > k:
        newOffspring1 = mutate1offspring(offsprings[0])
        fitNewOffspring1 = calcmutationFitness(newOffspring1)

    fitNewOffspring2 = calcmutationFitness(newOffspring2)
    while fitNewOffspring2 > k:
        newOffspring2 = mutate1offspring(offsprings[1])
        fitNewOffspring2 = calcmutationFitness(newOffspring2)

    return newOffspring1, newOffspring2

def newPopFitness(pop_size, newPopulation): #to print total value and test case id
    fitness = np.zeros(pop_size) #carry fitness of each testcase
    bestFitness = 0
    bestFitnessId = 0 #testcase ID
    f = 0
    for chromosome in newPopulation:
        sum = 0
        w = 0
        for gene in chromosome:
            sum += gene * values[w]
            w += 1
        fitness[f] = sum
        if sum > bestFitness:
            bestFitness = sum
            bestFitnessId = f
        f += 1
    print("The total value is: ",bestFitness)

    return bestFitnessId

def output(bestChromosome, newPopulation, weights, values): #print the total value and value and weight of each item
    chromosome = newPopulation[bestChromosome]
    numberOfItems = np.sum(chromosome)
    print("Number of Items ", numberOfItems)
    for i in range(len(weights)):
        if chromosome[i] == 1:
            print(weights[i], " ", values[i])


def mergePopulation(PopulationFitness, newPopulationFitness,pop_size):
    newFitness=np.zeros(pop_size)
    for i in range(pop_size):
        if PopulationFitness[i] > newPopulationFitness[i]:
            newFitness[i]=i
        else:
            newFitness[i] = i+pop_size

    return newFitness

def Replacement(Population, newPopulation,newFitness, pop_size):
    bestNewPopulation = np.zeros([pop_size, N_items])
    for i in range(pop_size):
            if newFitness[i]<100:
                PopulationIndex = int(newFitness[i])
                bestNewPopulation[i]=Population[PopulationIndex]
            else:
                newPopulationIndex=int(newFitness[i]-100)
                bestNewPopulation[i] = newPopulation[newPopulationIndex]
    return bestNewPopulation

with open('D:/Fourth Level/ga assignment 1/knapsack_input.txt', 'r') as fname:
    inputs = fname.readlines()
    cleanInput=[]
    for x in inputs:
        val = x.strip()
        cleanInput.append(val)

C = int(cleanInput[0])
cleanInputindex=1
pop_size = 100
for T in range(C):
    while cleanInput[cleanInputindex]=='':
        cleanInputindex+=1
    N_items = int(cleanInput[cleanInputindex])
    cleanInputindex+=1
    Size_k = int(cleanInput[cleanInputindex])
    cleanInputindex+=1
    weight = []
    values = []
    for items in range(N_items):
        w,b=map(int,cleanInput[cleanInputindex].split())
        weight.append(w)
        values.append(b)
        cleanInputindex+=1
    Population = InitializePop(pop_size, N_items, Size_k, weight)
    for itra in range(1000):
        newPopulation = np.zeros([pop_size, N_items])
        PopulationFitness = calcFitness(Population, values, pop_size)
        for i in range(50):
            ch = selection(PopulationFitness)
            offSprings = crossover(ch)
            MutatedCh = mutation(offSprings, Size_k)
            newPopulation[ch[0]] = MutatedCh[0]
            newPopulation[ch[1]] = MutatedCh[1]
        newPopulationFitness = calcFitness(newPopulation, values, pop_size)
        newFitenese=mergePopulation(PopulationFitness,newPopulationFitness,pop_size)
        Population=Replacement(Population,newPopulation,newFitenese,pop_size)
        #print(Population)

    print("Case ",T+1)
    bestChromosome = newPopFitness(pop_size, Population)
    output(bestChromosome, Population, weight, values)


