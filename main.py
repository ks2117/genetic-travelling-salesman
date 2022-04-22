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
            self.distances = [[int(np.linalg.norm([self.coordinates[i], self.coordinates[j]])) for i in range(self.number_of_cities)] for j in range(self.number_of_cities)]
        self.city_names = ["X"] + [chr(65 + i) for i in range(self.number_of_cities)]
        self.population_size = population_size
        self.number_of_generations = number_of_generations

    def partially_mapped_crossover(self, path1, path2):
        # len(self.distances) - 1 is the number of cities in the array
        new_path1 = np.zeros(self.number_of_cities)
        new_path2 = np.zeros(self.number_of_cities)

        subsequence_length = random.randint(0, self.number_of_cities)
        # TODO: consider normal distribution for above

        first_city = random.randint(0, self.number_of_cities - subsequence_length)

        new_path1[first_city:first_city + subsequence_length] = path1[first_city:first_city + subsequence_length]
        new_path2[first_city:first_city + subsequence_length] = path2[first_city:first_city + subsequence_length]

        counter1 = 0
        counter2 = 0

        for i in range(self.number_of_cities):
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

    def roulette_wheel_selection(self, generation):
        weights = generation_fitness(generation)
        weights = weights / sum(weights)
        pairings = [np.random.choice([i for i in range(self.population_size)], size=2, replace=False, p=weights) for _ in range(self.population_size // 2)]
        return [[generation[p1], generation[p2]] for p1, p2 in pairings]
        # return [np.random.choice(generation, size=2, replace=False, p=weights) for _ in range(self.population_size // 2)]

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
            #TODO MUTATION
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
best_path, history = g.train()
# print(best_path)
plt.plot(history)
plt.show()
# g.display_cities()