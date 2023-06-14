
from typing import Callable
import numpy as np

class Individual:

    def __init__(self, set_size: int, amount_selections: int, genotype: np.ndarray):
        if genotype is None:
            self.genotype = np.array([1]*amount_selections + [0]*(set_size-amount_selections))
            np.random.shuffle(self.genotype)
        else:
            assert np.sum(genotype) == amount_selections
            assert len(genotype) == set_size
            self.genotype = genotype
        


class FitnessBudgetExceeded(Exception):
    pass

class FitnessFunction:
    
    evaluations = 0
    
    def __init__(self, obj_func: Callable, evaluation_budget: int = 1e6):
        self.obj_func = obj_func
        self.evaluation_budget = evaluation_budget

    def evaluate(self, individual: Individual):
        self.evaluations += 1
        return self.obj_func(individual.genotype)
    



class GeneticAlgorithm:

    def __init__(self, population_size: int, set_size: int, amount_selections: int, fitness_function: FitnessFunction):
        self.set_size = set_size
        self.amount_selections = amount_selections
        self.population_size = population_size
        self.population = [Individual(set_size, amount_selections) for _ in range(self.population_size)]
        self.fitness_function = fitness_function

    def run(self):
        while self.fitness_function.evaluations < self.fitness_function.evaluation_budget:
            self.create_offspring()
            sorted_population = sorted(self.population, key=lambda x: -self.fitness_function.evaluate(x))
            self.population = sorted_population[:self.population_size]
        sorted_population = sorted(self.population, key=lambda x: -self.fitness_function.evaluate(x))
        return sorted_population[0], self.fitness_function.evaluate(sorted_population[0])

    def create_offspring(self):
        offspring = []
        for individual in self.population:
            selected_genes = np.where(individual.genotype == 1)[0][0]
            not_selected_genes = np.where(~individual.genotype == 0)[0][0]
            unselect_gene = np.random.randint(0,len(selected_genes))
            select_gene = np.random.randint(0,len(not_selected_genes))
            new_genotype = individual.genotype.copy()
            new_genotype[unselect_gene] = 0
            new_genotype[select_gene] = 1
            offspring.append(Individual(set_size=self.set_size, amount_selections=self.amount_selections, genotype=new_genotype))
        self.population += offspring
