import random

import numpy as np
import matplotlib.pyplot as plt


def generation_fitness(generation):
    return np.array([1 / np.sum(path) for path in generation])


class GeneticTrainer:

    def __init__(self, distances=None, population_size=100, number_of_generations=10):
        if distances is not None:
            self.distances = distances
            self.number_of_cities = len(distances)
        else:
            self.number_of_cities = 20
            self.coordinates = [[100.0, 150.0]] + [[random.random() * 275, random.random() * 200] for i in range(20)]
            self.distances = [
                [int(np.linalg.norm([self.coordinates[i], self.coordinates[j]])) for i in range(self.number_of_cities)]
                for j in range(self.number_of_cities)]
        self.city_names = ["X"] + [chr(65 + i) for i in range(self.number_of_cities)]
        self.population_size = population_size
        self.number_of_generations = number_of_generations

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


    def roulette_wheel_selection(self, generation):
        probabilities = generation_fitness(generation)
        probabilities = probabilities / sum(probabilities)
        pairings = [np.random.choice([i for i in range(self.population_size)], size=2, replace=False, p=probabilities)
                    for _ in range(self.population_size // 2)]
        return [[generation[p1], generation[p2]] for p1, p2 in pairings]

    def train(self):
        cities = np.array([i + 1 for i in range(self.number_of_cities)])
        generation = [np.random.permutation(cities) for _ in range(self.population_size)]
        generation_counter = 1
        history = np.zeros(self.number_of_generations + 1)
        history[0] = max(generation_fitness(generation))
        while True:
            if generation_counter > self.number_of_generations:
                break
            pairings = self.roulette_wheel_selection(generation)
            generation = [self.partially_mapped_crossover(path1, path2) for path1, path2 in pairings]
            generation = [item for sublist in generation for item in sublist]
            fitness = generation_fitness(generation)
            history[generation_counter] = max(fitness)
            generation_counter += 1
            # TODO MUTATION
        fitness = generation_fitness(generation)
        return generation[np.argmax(fitness)], history

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


g = GeneticTrainer(number_of_generations=100)
test_p1 = np.array([ord(c) - 64 for c in "JBFCADHGIE"])
test_p2 = np.array([ord(c) - 64 for c in "FAGDHCEBJI"])
pmc1, pmc2 = g.partially_mapped_crossover(test_p1, test_p2, number_of_cities=10, distribution="test", first_city=2)
print(g.decode_cities(pmc1))
order_crossover1, order_crossover2 = g.order_crossover(test_p1, test_p2, number_of_cities=10, distribution="test",
                                                       first_city=1)
print(g.decode_cities(order_crossover1))
cycle_crossover1, cycle_crossover2 = g.cycle_crossover(test_p1, test_p2, number_of_cities=10)
print(g.decode_cities(cycle_crossover1))
# best_path, history = g.train()
# print(best_path)
# plt.plot(history)
# plt.show()
# g.display_cities()
