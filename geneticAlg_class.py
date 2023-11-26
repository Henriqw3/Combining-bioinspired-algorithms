import numpy as np
import pandas as pd
from perceptron_class import Perceptron

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate=0.1, crossover_rate=0.7, perceptron_epochs=50):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.perceptron_epochs = perceptron_epochs

    def initialize_population(self, num_weights):
        return np.random.rand(self.population_size, num_weights)

    def calculate_accuracy(self, perceptron, X, y):
        y_pred = perceptron.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def calculate_mse(self, perceptron, X, y):
        errors = y - perceptron.predict(X)
        mse = np.mean(errors**2)
        return mse

    def evaluate_fitness(self, population, X_train, y_train):
        fitness_scores = []

        for weights in population:
            perceptron = Perceptron(epochs=self.perceptron_epochs)
            perceptron.set_weights(weights)
            
            # Escolher precisão ou erro médio quadrático
            #fitness = self.calculate_accuracy(perceptron, X_train, y_train)
            fitness = -self.calculate_mse(perceptron, X_train, y_train)  # Use -mse para minimização

            fitness_scores.append(fitness)

        return np.array(fitness_scores)

    def select_parents(self, population, fitness_scores):

        probabilities = fitness_scores / np.sum(fitness_scores)
        
        parents_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        
        return parents_indices


    def crossover(self, parents, crossover_rate=0.7):
        descendants = np.empty_like(parents)
        crossover_points = np.random.randint(1, len(parents[0]), size=len(parents))

        for i in range(0, len(parents) - 1, 2):
            if np.random.rand() < crossover_rate:
                crossover_point = crossover_points[i]
                descendants[i, :crossover_point] = parents[i, :crossover_point]
                descendants[i, crossover_point:] = parents[i + 1, crossover_point:]
                descendants[i + 1, :crossover_point] = parents[i + 1, :crossover_point]
                descendants[i + 1, crossover_point:] = parents[i, crossover_point:]
            else:
                descendants[i, :] = parents[i, :]
                descendants[i + 1, :] = parents[i + 1, :]

        return descendants



    def mutate(self, population, mutation_rate=0.1):
        mutated_population = population.copy()
        for i in range(len(population)):
            for j in range(len(population[i])):
                if np.random.rand() < mutation_rate:
                    mutated_population[i, j] = np.random.rand()
        return mutated_population


    def run(self, X_train, y_train, ga_epochs):
        num_weights = X_train.shape[1] + 1
        population = self.initialize_population(num_weights)

        for generation in range(ga_epochs):
            fitness_scores = self.evaluate_fitness(population, X_train, y_train)
            parents = self.select_parents(population, fitness_scores)
            descendants = self.crossover(population[parents])
            descendants = self.mutate(descendants)

            population[:len(descendants)] = descendants


        best_index = np.argmax(self.evaluate_fitness(population, X_train, y_train))
        best_weights = population[best_index]
        return best_weights
