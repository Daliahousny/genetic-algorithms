
from itertools import chain
from operator import itemgetter
import pandas as pd
import numpy as np
import random
import math

Population = []
def intializePopulation(chromosomes, coefficients):
    Population = np.zeros([chromosomes, coefficients])
    i = 0
    while (i < chromosomes):
        j = 0
        while (j < coefficients):
            Population[i][j] = round(random.uniform(-10,10),2)
            j += 1
        i += 1

    Population = [l.tolist() for l in Population]

    return Population
def calculateFitness(Population, N):
    mse = []
    coefficient = 0
    for chromosome in Population:
        total_error = 0
        for point in N:
            for coefficient in range(len(chromosome)):
                y_predicted = chromosome[coefficient] * math.pow(point[0], coefficient)
                total_error += math.pow(y_predicted - point[1], 2)
        MSE = total_error/len(N)
        mse.append(MSE)
    return mse
def selectionfun(Population,solutions,N):
    minfit_index = []
    selected = 0
    for i in range(solutions):
      # selectionlist = []
      random_chrom = []
      m = []
      index_value = random.sample(list(enumerate(Population)), 6)
      #print(index_value[1])
      for j in index_value:
        random_chrom.append(j[1])
      m.extend(calculateFitness(random_chrom, N))
      minfit=min(m)
      minfit_index.append(m.index(minfit))
      selected = list(itemgetter(*minfit_index)(random_chrom))
    return selected
def crossover(selectedchrom):
    offsprings = []
    offsprings2 = []
    offspring1 = []
    offspring2 = []
    parent1 = selectedchrom[0]
    parent2 = selectedchrom[1]
    crossPoint1 = random.randint(1, int(len(parent1) / 2))
    crossPoint2 = random.randint(int(len(parent1) / 2), len(parent1) - 1)
    rc = random.uniform(0.0, 1.0)
    pc = random.uniform(0.4, 0.7)
    if rc < pc:
        for i in range(crossPoint1):
            offspring1.append(parent1[i])
        for i in range(crossPoint1, crossPoint2):
            offspring1.append(parent2[i])
        for i in range(crossPoint2, len(parent1)):
            offspring1.append(parent1[i])
        for i in range(crossPoint1):
            offspring2.append(parent2[i])
        for i in range(crossPoint1, crossPoint2):
            offspring2.append(parent1[i])
        for i in range(crossPoint2, len(parent2)):
            offspring2.append(parent2[i])
        offsprings = offspring1
        offsprings2 = offspring2
        return offsprings, offsprings2
    else:
         offsprings.append(parent1)
         offsprings.append(parent2)
    return offsprings

def mutation(offsprings):
    i=0
    T=100
    t=0
    b=random.uniform(0.5,5)
    y=0
    delta=0
    for offspring in offsprings:
        for bit in range(len(offspring)):
            rm=random.uniform(0,1)
            pm=random.uniform(0.001,0.1)
            if(rm<=pm):
                lower_bound=offspring[bit]-(-10)
                upper_bound=10-offspring[bit]
                r1=random.uniform(0,1)
                if (r1<=0.5):
                   y=lower_bound
                else:
                    y=upper_bound
                k=((1-(t/T))**b)
                delta=y*(1-(r1**k))
                if (y==lower_bound):
                    xnew=offspring[bit]-delta
                    offspring[bit] = xnew
                else:
                    ynew=offspring[bit]+delta
                    offspring[bit] = ynew
    return offsprings

def elitistfun(Population,offsprings,N):
    replaced = []
    templist = []
    ind = []
    tmp = []
    templist.extend(Population)
    templist.extend(offsprings)
    #print(f"all {templist}")
    tmp.append(calculateFitness(templist,N))
    flatten_list = [j for sub in tmp for j in sub]
    Sorted = Sorted = sorted(flatten_list, key = lambda x:float(x))
    #print(flatten_list)
    for s in Sorted:
        ind.append(flatten_list.index(s))
    print(ind)
    for i in range(len(Population)):
          replaced.append(list(itemgetter(*ind)(templist)))
    replaced= [j for sub in replaced for j in sub]
    return replaced
with open("D:\\Fourth Level\\ga assignment 2\\curve_fitting_input.txt", 'r') as fname:
    inputs = fname.readlines()
    listlist=[]
    pop_size=100
    for x in inputs:
        value = x.replace('\n', "").split()
        listlist.append(value)
    n_samples = int(listlist[0][0])
    listlist.pop(0)
d = []
xy = []
for i in listlist:
    if i[0] == '64':
        d.append(i)
    else:
        xy.append(i)
degrees = []
for i in range(len(d)):
    degree = list(map(int, d[i]))
    degrees.append(degree)

coefficients=[]
j=0
for j in range(len(degrees)):
     co=degrees[j][1]+1
     coefficients.append(co)
     j+=1
     for l in range(len(degrees)):
         print(l)
x_and_y = []
for i in range(len(xy)):
    xs_ys = list(map(float, xy[i]))
    x_and_y.append(xs_ys)

Population = intializePopulation(pop_size,co)
# k=0
# for k in range(len(Population):
#       print(Population[k])
#       k+=1
for j in range(1000):
        newPopulation = np.zeros([pop_size, co])
         # k = 0
         # for k in range(len(Population):
         #    print(newPopulation[k])
         #    k += 1
        PopulationFitness = calculateFitness(Population,x_and_y)
        print('mse',PopulationFitness)
        for i in range(50):
            chrom = selectionfun(Population,2,x_and_y)
            offSprings = crossover(chrom)
            MCh = mutation(offSprings)
            #newPopulation=[]
            # newPopulation[0] .append(MCh[0])
            # newPopulation[1] .append(MCh[1])
            Population = elitistfun(Population, MCh, x_and_y)
#print('dataset index', j+1)
        with open("E:/New folder", 'w') as fname:
            inputs = fname.writelines()
            listlistlist = []
            for x in inputs:
                listlistlist.append( PopulationFitness)
                #listlistlist.append(Population[k])
                listlistlist.append(l)