import csv
import random
from copy import deepcopy as dc
import numpy as np
import show_image
import matplotlib.pyplot as plt


ADJ_PATH = "adjMatrix.csv"
adjMatrix = np.loadtxt(ADJ_PATH, delimiter=",")
COLORS = ['G', 'R', 'B', 'L']
NUM_OF_NODES = 12
FILE = "Run.csv"


class Engine:

    def __init__(self, graph, parameters):
        self.graph = graph
        self.population_size = parameters['population_size']
        self.top_solutions_probability = int(parameters['top_solutions_probability'] * self.population_size)
        self.mutation_probability = int(parameters['mutation_probability'] * self.population_size)
        self.crossover_probability = int(parameters['crossover_probability'] * self.population_size)
        self.num_of_vertex = parameters['num vertex']
        self.num_iter = parameters['number_of_iterations']

    def run(self):
        """ Run the Engine methods:
        # initialize the population
        #reproduce
        #crossover
        #mutate
        show the best solution image
        """
        csv_file = open(FILE, 'w')
        writer = csv.writer(csv_file, dialect='excel')
        random.seed()
        avg = []
        best = []
        worst = []
        population = self._initialize_population()
        for i in range(self.num_iter):
            avg_fitness, best_fitness, worst_fitness, top_solutions, rest_population = self._pick_best(population)
            new_crossovers = self.crossover(rest_population)
            population = top_solutions + new_crossovers
            self.mutate(population)
            print('generation: {} avg: {} best: {} worst: {}'.format(i, avg_fitness, best_fitness[1], worst_fitness[1]))
            status = (i, avg_fitness, best_fitness[1], worst_fitness[1])
            writer.writerow(status)
            avg.append(avg_fitness)
            best.append(best_fitness[1])
            worst.append(worst_fitness[1])
            colors_map = show_image.set_coloring_for_image(best_fitness)
            show_image.image(colors_map)
            if best_fitness[1] == 0:
                break

        self.plot(avg, best, worst)

    def plot(self, avg, best, worst):
        ''' Ploting of the genetic algorithm '''
        plt.plot(avg, label='avg')
        plt.plot(best, label='best')
        plt.plot(worst, label='worst')
        plt.ylim(0, self.num_iter)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Genetic Algorithm Performance')
        plt.legend()
        plt.show()
        exit(0)


    def mutate(self, population):
        """
        for specific probability mutate P random genes.
        :param population: the current population
        :return: mutate genes in population
        """
        for i in range(self.mutation_probability):
            gene = random.choice(population)
            rand_vertex = random.choice(list(gene[0].keys()))
            gene[0][rand_vertex] = random.choice(COLORS)
            fitness = self._calc_fitness(gene[0])
            new_gene = (gene[0], fitness)
            population[population.index(gene)] = new_gene

    def crossover(self, population):
        """
        Crossover between two genes in population.
        Pick random slicing of the gene1 and replace with gene2
        :param population: population: the current population
        :return: crossover in population
        """
        new_population = []
        for gene1 in population:
            gene2 = random.choice(population[0:int(self.population_size/2)])
            cutoff = random.randint(0, self.num_of_vertex)
            coloring = {}
            count = 0
            for v, color in gene1[0].items():
                if count == cutoff:
                    break
                coloring[v] = color
                count += 1
            count = 0
            for v, color in gene2[0].items():
                count += 1
                if count < cutoff:
                    continue
                coloring[v] = color
            fitness = self._calc_fitness(coloring)
            new_population.append((coloring, fitness))
        return new_population

    def _initialize_population(self):
        """
        Initializing population_size of Nodes with random colors
        and calculation of fitness for each individual
        :return: list of tuples (coloring, fitness) where coloring = {vertex: color}
        """
        random.seed()
        population = []
        while len(population) < self.population_size:
            coloring = {}
            for v in self.graph.nodes.values():
                chosen_color = random.choice(COLORS)
                coloring[v] = chosen_color
            fitness = self._calc_fitness(coloring)
            population.append((coloring, fitness))
        return population

    def _pick_best(self, population):
        """
        Pick the best solution according to fitness and slicing the population for reproduce
        :param population: current population
        :return: average fitness of population, best solution in current population, top solutions, all others.
        """
        population.sort(key=lambda x: x[1])
        fitnesses = [f[1] for f in population]
        avg = sum(fitnesses) / len(fitnesses)
        top_solutions_rate = self.top_solutions_probability
        top_solutions = population[0:top_solutions_rate]
        rest = population[top_solutions_rate:]
        return avg, population[0], population[-1], top_solutions, rest

    def _calc_fitness(self, coloring):
        """
        Calculation of the current fitness of each coloring based on sef fitness of each node
        :param coloring: dict {vertex : color)
        :return: int fitness
        """
        fitness = 0
        for v, color in coloring.items():
            fitness += v.self_fitness(coloring, color)
        return fitness


class Graph:
    def __init__(self, adjMatrix):
        self.adjMatrix = adjMatrix
        self.nodes = {}
        self.build_nodes()


    def build_nodes(self):
        """
        Build the graph nodes with neighbours according to adj Matrix
        :return:
        """
        for i in range(NUM_OF_NODES):
            result = np.where(self.adjMatrix[i, :] == 1)
            n = np.add(result, 1)
            node_i = Node(ID=i + 1, neighbours=n)
            self.nodes[i + 1] = dc(node_i)


class Node:

    def __init__(self, ID, neighbours):
        self.ID = ID
        self.neighbours_index = neighbours[0]

    def self_fitness(self, coloring, this_color):
        """
        Check for every node in coloring if it's a neighbour of this node and
        check if they have same color, if do- count.
        :param coloring: dict {node : color}
        :param this_color:
        :return: int self fitness
        """
        count = 0
        for v, color in coloring.items():
            if v.ID in self.neighbours_index:
                if color == this_color:
                    count += 1
        return count

    def __hash(self):
        return self.ID


params = {
    'population_size': 100,
    'mutation_probability': 0.2,
    'crossover_probability': 0.95,
    'num vertex': NUM_OF_NODES,
    'number_of_iterations': 100,
    'top_solutions_probability': 0.05
}


def main():
    graph = Graph(adjMatrix)
    engine = Engine(graph, params)
    engine.run()


if __name__ == '__main__':
    main()
