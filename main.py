import random

import numpy as np
import matplotlib.pyplot as plt


# TODO: simulating annealing


class GeneticTrainer:

    def __init__(self, distances=None, population_size=100, number_of_generations=10, mutation_rate=0.1):
        if distances is not None:
            self.distances = distances
            self.number_of_cities = len(distances) - 1
        else:
            self.number_of_cities = 20
            self.coordinates = [[100, 150]] + [[int(random.random() * 275), int(random.random() * 200)] for _ in
                                               range(20)]
            self.distances = [
                [int(np.linalg.norm([self.coordinates[i], self.coordinates[j]])) for i in range(len(self.coordinates))]
                for j in range(len(self.coordinates))]
        self.city_names = ["X"] + [chr(65 + i) for i in range(self.number_of_cities)]
        self.population_size = population_size
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate

    def evaluate_path(self, path):
        path_length = 0
        current_city = 0
        for city in path:
            path_length += self.distances[current_city][int(city)]
            current_city = int(city)
        path_length += self.distances[current_city][0]
        return path_length

    def generation_fitness(self, generation, p=0, epsilon=1):
        evaluation = np.array([self.evaluate_path(path) for path in generation])
        evaluation = evaluation - np.min(evaluation) * (1 - p) + epsilon
        return 1 / evaluation

    def partially_mapped_crossover(self, path1, path2, number_of_cities=None, distribution="uniform", first_city=None):
        if number_of_cities is None:
            number_of_cities = self.number_of_cities
        # len(self.distances) - 1 is the number of cities in the array
        new_path1 = np.zeros(number_of_cities)
        new_path2 = np.zeros(number_of_cities)

        if distribution == "uniform":
            subsequence_length = random.randint(0, number_of_cities)
        elif distribution == "normal":
            subsequence_length = 5  # TODO: consider normal distribution for above
        elif distribution == "test":
            subsequence_length = 5  # TODO: consider normal distribution for above
        else:
            subsequence_length = number_of_cities / 2

        if first_city is None:
            first_city = random.randint(0, number_of_cities - subsequence_length)

        new_path1[first_city:first_city + subsequence_length] = path1[first_city:first_city + subsequence_length]
        new_path2[first_city:first_city + subsequence_length] = path2[first_city:first_city + subsequence_length]

        counter1 = 0
        counter2 = 0

        for i in range(number_of_cities):
            if first_city <= i < first_city + subsequence_length:
                continue
            else:
                while path2[counter1] in new_path1:
                    counter1 += 1
                new_path1[i] = path2[counter1]

                while path1[counter2] in new_path2:
                    counter2 += 1
                new_path2[i] = path1[counter2]
        return new_path1, new_path2

    def order_crossover(self, path1, path2, number_of_cities=None, distribution="uniform", first_city=None):
        if number_of_cities is None:
            number_of_cities = self.number_of_cities
        new_path1 = np.zeros(number_of_cities)
        new_path2 = np.zeros(number_of_cities)

        if distribution == "uniform":
            subsequence_length = random.randint(0, number_of_cities)
        elif distribution == "normal":
            subsequence_length = 5  # TODO: consider normal distribution for above
        elif distribution == "test":
            subsequence_length = 5  # TODO: consider normal distribution for above
        else:
            subsequence_length = number_of_cities / 2

        if first_city is None:
            first_city = random.randint(0, number_of_cities - subsequence_length)

        new_path1[first_city:first_city + subsequence_length] = path1[first_city:first_city + subsequence_length]
        new_path2[first_city:first_city + subsequence_length] = path2[first_city:first_city + subsequence_length]

        counter1 = first_city + subsequence_length
        counter2 = first_city + subsequence_length
        for i in range(subsequence_length, number_of_cities):
            while path2[counter1 % number_of_cities] in new_path1:
                counter1 += 1
            new_path1[(i + first_city) % number_of_cities] = path2[counter1 % number_of_cities]
            while path1[counter2 % number_of_cities] in new_path2:
                counter2 += 1
            new_path2[(i + first_city) % number_of_cities] = path1[counter2 % number_of_cities]
        return new_path1, new_path2

    def cycle_crossover(self, path1, path2, number_of_cities=None):
        if number_of_cities is None:
            number_of_cities = self.number_of_cities
        new_path1 = np.zeros(number_of_cities)
        new_path2 = np.zeros(number_of_cities)

        city = path1[0]
        index = 0
        while city not in new_path1:
            new_path1[index] = city
            index = np.where(path1 == path2[index])[0][0]
            city = path1[index]
        counter = 0
        for i in range(number_of_cities):
            if new_path1[i] == 0:
                while path2[counter] in new_path1:
                    counter += 1
                new_path1[i] = path2[counter]

        city = path2[0]
        index = 0
        while city not in new_path2:
            new_path2[index] = city
            index = np.where(path2 == path1[index])[0][0]
            city = path1[index]
        counter = 0
        for i in range(number_of_cities):
            if new_path2[i] == 0:
                while path1[counter] in new_path2:
                    counter += 1
                new_path2[i] = path1[counter]

        return new_path1, new_path2

    def roulette_wheel_selection(self, generation, weights):
        probabilities = weights / np.sum(weights)
        pairings = [np.random.choice([i for i in range(self.population_size)], size=2, replace=False, p=probabilities)
                    for _ in range(self.population_size // 2)]
        return [[generation[p1], generation[p2]] for p1, p2 in pairings]

    def stochastic_universal_sampling(self, generation, n, fitness):
        p = sum(fitness) / n
        indices = list(range(len(fitness)))
        indices.sort(key=fitness.__getitem__, reverse=True)
        sorted_fitness = list(map(fitness.__getitem__, indices))
        sorted_generation = list(map(generation.__getitem__, indices))
        partial_sums = np.array([f for f in sorted_fitness])
        for i in range(1, len(fitness)):
            partial_sums[i] += partial_sums[i-1]
        start = random.random() * p
        pointers = np.array([start + i * p for i in range(n)])
        keep = [0 for _ in range(n)]
        counter = 0
        for point in pointers:
            i = 0
            while partial_sums[i] < point:
                i += 1
            keep[counter] = sorted_generation[i]
            counter += 1
        pairings = np.array([np.random.choice([i for i in range(n)], size=2, replace=False) for _ in range(self.population_size // 2)])
        return [[generation[p1], generation[p2]] for p1, p2 in pairings]

    def tournament_selection(self, generation, fitness, tournament_size, p):
        n = len(fitness)
        indices = list(range(n))
        indices.sort(key=fitness.__getitem__, reverse=True)
        sorted_generation = list(map(generation.__getitem__, indices))
        probabilities = np.array([p*(1-p) ** i for i in range(tournament_size)])
        probabilities = probabilities / sum(probabilities)
        sorted_truncated_generation = sorted_generation[:tournament_size]
        pairings = [np.random.choice([i for i in range(len(sorted_truncated_generation))], size=2, replace=False, p=probabilities)
                    for _ in range(self.population_size // 2)]
        return [[sorted_truncated_generation[p1], sorted_truncated_generation[p2]] for p1, p2 in pairings]

    def truncation_selection(self, generation, fitness, p):
        n = len(fitness)
        indices = list(range(n))
        indices.sort(key=fitness.__getitem__, reverse=True)
        sorted_generation = list(map(generation.__getitem__, indices))
        truncated_generation = sorted_generation[:int(n*p)]
        truncated_generation = np.repeat(truncated_generation, int(1/p), axis=0)
        pairings = [np.random.choice([i for i in range(len(truncated_generation))], size=2, replace=False)
                    for _ in range(self.population_size // 2)]
        return [[truncated_generation[p1], truncated_generation[p2]] for p1, p2 in pairings]

    def mutate(self, path):
        length = len(path)
        mutations = int(random.random() * self.mutation_rate * length)
        for _ in range(mutations):
            i1 = random.randint(0, length - 1)
            i2 = (i1 + random.randint(0, length - 2)) % length
            path[i1], path[i2] = path[i2], path[i1]
        return path

    def train(self, selection_method="rws", crossover_method="pmc"):
        cities = np.array([i + 1 for i in range(self.number_of_cities)])
        generation = [np.random.permutation(cities) for _ in range(self.population_size)]
        generation_counter = 1
        history = np.zeros(self.number_of_generations + 1)
        fitness = self.generation_fitness(generation)
        best_path = generation[np.argmax(fitness)]
        history[0] = self.evaluate_path(best_path)
        while True:
            if generation_counter > self.number_of_generations:
                break
            fitness = self.generation_fitness(generation)
            if selection_method == "sus":
                pairings = self.stochastic_universal_sampling(generation, int(0.5 * self.population_size), fitness)
            elif selection_method == "tournament":
                pairings = self.tournament_selection(generation, fitness, int(0.2 * self.population_size), p=4/9)
            elif selection_method == "truncation":
                pairings = self.truncation_selection(generation, fitness, p=0.2)
            else:
                pairings = self.roulette_wheel_selection(generation, fitness)

            if crossover_method == "order":
                generation = [self.order_crossover(path1, path2) for path1, path2 in pairings]
            elif crossover_method == "cycle":
                generation = [self.cycle_crossover(path1, path2) for path1, path2 in pairings]
            else:
                generation = [self.partially_mapped_crossover(path1, path2) for path1, path2 in pairings]
            generation = [item for sublist in generation for item in sublist]
            generation = [self.mutate(population) for population in generation]
            fitness = self.generation_fitness(generation)
            best_path = generation[np.argmax(fitness)]
            history[generation_counter] = self.evaluate_path(best_path)
            generation_counter += 1
        return best_path, history

    def display_cities(self):
        x = [self.coordinates[i][0] for i in range(len(self.coordinates))]
        y = [self.coordinates[i][1] for i in range(len(self.coordinates))]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        for i in range(len(x)):
            ax.annotate(self.city_names[i], (x[i] + 2, y[i] + 2))
        plt.show()

    def decode_cities(self, path):
        return [chr(int(city) + 64) for city in path]


def test():
    g = GeneticTrainer()
    test_p1 = np.array([ord(c) - 64 for c in "JBFCADHGIE"])
    test_p2 = np.array([ord(c) - 64 for c in "FAGDHCEBJI"])
    pmc1, pmc2 = g.partially_mapped_crossover(test_p1, test_p2, number_of_cities=10, distribution="test", first_city=2)
    print(g.decode_cities(pmc1))
    order_crossover1, order_crossover2 = g.order_crossover(test_p1, test_p2, number_of_cities=10, distribution="test",
                                                           first_city=1)
    print(g.decode_cities(order_crossover1))
    cycle_crossover1, cycle_crossover2 = g.cycle_crossover(test_p1, test_p2, number_of_cities=10)
    print(g.decode_cities(cycle_crossover1))


test()
g = GeneticTrainer(population_size=100, number_of_generations=500, mutation_rate=0)
best_path, history = g.train(crossover_method="pmc", selection_method="truncation")
print(best_path)
plt.plot(history)
plt.show()
best_path, history = g.train(crossover_method="pmc", selection_method="rws")
print(best_path)
plt.plot(history)
plt.show()
g.display_cities()
