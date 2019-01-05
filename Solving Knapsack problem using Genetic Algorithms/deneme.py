import random
import numpy
import matplotlib.pyplot as plt

def create_population(num):
    """
    gets a number and creates chromosomes in that number
    :param num: POPSIZE
    :return:
    """
    return [create_chromosome() for x in range(0, num)]


def create_chromosome():
    """
    creates chromosomes, each have genes number of items,
    genes are randomly generated 1 or 0
    :return:
    """
    return [random.randint(0, 1) for x in range(0, len(items))]


def calculate_fitness(chr):
    """
    gets a chromosome as parameter then returns it's fitness value as total value of items
    if chromosome have weight bigger then knapsack size, returns 0
    :param chr: a chromosome
    :return: fitness value of chromosome
    """
    total_value = 0
    total_weight = 0
    num = 0
    for gene in chr:
        if gene == 1:
            total_weight += items[num][0]
            total_value += items[num][1]
        num += 1
    if total_weight > knapsack_size:
        return 0
    else:
        return total_value


def select_parent(pop, f_list):
    """
    applies roulette wheel selection.
    :param pop: a population to select parents
    :param f_list: list of fitness values of that population
    :return: selected parent chromosome
    """
    max_fitness = sum(f_list)
    fixed_point = random.randint(0, max_fitness)
    current_val = 0
    for chr, ftns in zip(pop, f_list):
        current_val += ftns
        if current_val > fixed_point:
            return chr


def crossover(father, mother, n):
    """
    gets father and mother chromosomes then uses crossover prob. to apply crossover
    :param father: a chromosome
    :param mother: a chromosome
    :param n: point number of crossover
    :return: childrens list
    """
    gene_len = len(father) // n
    gene_len_list = []
    for i in range(len(father) // gene_len):
        gene_len_list.append(gene_len)
    if len(father) % gene_len != 0:
        gene_len_list[n - 1] = gene_len_list[n - 1] + (len(father) % gene_len)

    child_1 = []
    child_2 = []
    count = 0
    zipped = []
    for i in range(len(father)):
        zipped.append([father[i], mother[i]])

    for i, b in zip(gene_len_list, range(len(gene_len_list))):
        if b % 2 == 0:
            for a in range(i):
                child_1.append(zipped[count][0])
                child_2.append(zipped[count][1])
                count = count + 1
        else:
            for a in range(i):
                child_1.append(zipped[count][1])
                child_2.append(zipped[count][0])
                count = count + 1

    children = [child_1, child_2]
    return children


def mutate(children_pool):
    """
    gets children pool, then uses mutation prob. to apply mutation
    :param children_pool: gets all children
    :return: new children pool with mutated children
    """
    mutation_chance = MUTPROB
    new_children_pool = []
    for child in children_pool:
        if mutation_chance > random.random():
            r = random.randint(0, len(child)-1)
            if child[r] == 1:
                child[r] = 0
            else:
                child[r] = 1
        new_children_pool.append(child)
    return new_children_pool


def elitism(pop, ftns_list):
    """
    :param pop: a population
    :param ftns_list: fitness list of that population
    :return: best fit chromosome in that population
    """
    max_ftns = max(ftns_list)
    index_of = ftns_list.index(max_ftns)
    return pop[index_of]


# main code starts here #

f = open("items.txt", "r")
items = []
knapsack_size = 0
index = 0

for line in f:
    if index < 50:
        a = line.rstrip().split(" ")
        a[0] = int(a[0])
        a[1] = int(a[1])
        items.append(a)

    else:
        knapsack_size = int(line)
    index = index + 1

ITEMS_COUNT = len(items)

POPSIZE = int(input("Please enter population size: "))
GENNUM = int(input("Please enter generation number:"))
CROSPROB = float(input("Please enter crossover probability:"))  # float for between 0 and 1
MUTPROB = float(input("Please enter mutation probability:"))  # float for between 0 and 1

population = create_population(POPSIZE)  # initial population created
average_fitness_list=[]
for gen in range(0, GENNUM):  # our population is going to evolve GENNUM times

    fitness_list = []
    for chromosome in population:
        fitness = calculate_fitness(chromosome)
        fitness_list.append(fitness)

    best = elitism(population, fitness_list)

    parent_pool = []
    for i in range(len(population)):
        parent_pool.append(select_parent(population, fitness_list))  # selects a parent with roulette wheel selection

    if POPSIZE % 2 == 1:  # we need even number of parents for crossing over, get rid of random chromosome
        del parent_pool[random.randint(0, POPSIZE-1)]

    children_pool = []
    for father, mother in zip(parent_pool[0::2], parent_pool[1::2]):
        if CROSPROB > random.random():
            children = crossover(father, mother, 3)  # crossover parents with n points that given to the function
            children_pool.append(children[0])
            children_pool.append(children[1])
        else:
            children_pool.append(father)
            children_pool.append(mother)

    mutated_population = mutate(children_pool)

    if POPSIZE % 2 == 0:  # to get a room for best chromosome, get rid of random chromosome
        del mutated_population[random.randint(0, POPSIZE-1)]

    mutated_population.append(best)
    population = mutated_population
    average_fitness_list.append(numpy.mean(fitness_list))

best_population = population

x = [i for i in range(GENNUM)]
y = average_fitness_list
plt.plot(x, y, x, y, "ko")
plt.xlabel("generations")
plt.ylabel("avarage finess")
title = "pop:", POPSIZE, " gen:", GENNUM, "cp:", CROSPROB, "mp:", MUTPROB
plt.title(title)
plt.show()
