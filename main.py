import random

import numpy as np
import matplotlib.pyplot as plt


# TODO: simulating annealing


class GeneticTrainer:

    def __init__(self, distances=None, population_size=100, number_of_generations=10, mutation_rate=0.1):
        if distances is not None:
            self.distances = distances
            self.number_of_cities = len(distances)
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
        sum = 0
        current_city = 0
        for city in path:
            sum += self.distances[current_city][int(city)]
            current_city = int(city)
        sum += self.distances[current_city][0]
        return sum

    def generation_fitness(self, generation, p=0, epsilon=1):
        # p is a hyperparameter, when p is 0 the shortest path will have maximum fitness, when p=1 all elements will
        # have equal fitness. Low values of p correspond with giving higher fitness to shorter paths.
        # epsilon is a parameter to add numerical stability for cases where we want to use p==0
        evaluation = np.array([self.evaluate_path(path) for path in generation])
        if p == 0:
            best_fitness = np.min(evaluation)
            for i in range(len(evaluation)):
                if evaluation[i] == best_fitness:
                    evaluation[i] = 1
                else:
                    evaluation[i] = 0
                return evaluation
        else:
            evaluation = evaluation - (np.min(evaluation) + epsilon) * (1 - p)
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
        for i in range(first_city + subsequence_length, number_of_cities + 1):
            while path2[counter1] in new_path1:
                counter1 += 1
                counter1 = counter1 % number_of_cities
            new_path1[i % number_of_cities] = path2[counter1]
            while path1[counter2] in new_path2:
                counter2 += 1
                counter2 = counter2 % number_of_cities
            new_path2[i % number_of_cities] = path1[counter2]
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

    def stochastic_universal_sampling(self, generation, number_of_offspring, fitness, p):
        n = len(fitness)
        indices = range(n)
        indices.sort(key=fitness.__getitem__)
        sorted_fitness, sorted_generation = (np.array(t) for t in zip(*sorted(zip(fitness, generation))))
        partial_sums = np.array([f for f in sorted_fitness])
        for i in range(1, n):
            partial_sums[i] += partial_sums[i-1]
        start = random.random() * p
        pointers = np.array([start + i * p for i in range(number_of_offspring-2)])
        keep = [0 for _ in range(number_of_offspring)]
        counter = 0
        for p in pointers:
            i = 0
            while partial_sums[i] < p:
                i += 1
            keep[counter] = sorted_generation[i]
            counter += 1
        return keep

    def tournament_selection(self, generation):
        # TODO
        pass

    def truncation_selection(self, generation):
        # TODO
        pass

    def mutate(self, path):
        length = len(path)
        mutations = int(random.random() * self.mutation_rate * length)
        for _ in range(mutations):
            i1 = random.randint(0, length - 1)
            i2 = (i1 + random.randint(0, length - 2)) % length
            path[i1], path[i2] = path[i2], path[i1]
        return path

    def train(self):
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
            pairings = self.roulette_wheel_selection(generation, fitness)
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


g = GeneticTrainer(population_size=50, number_of_generations=1000, mutation_rate=0.2)
best_path, history = g.train()
print(best_path)
plt.plot(history)
plt.show()
g.display_cities()
