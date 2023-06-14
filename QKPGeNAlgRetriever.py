'''Roberta embeddings Retriever'''

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import Optional, Callable

from accelerate import Accelerator
from sentence_transformers import SentenceTransformer

import numpy as np

from transformers import pipeline

from sklearn.metrics.pairwise import cosine_similarity

logger = get_logger(__name__)


class QPKRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class
        Class of Random Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        seed (`int`, optional): Seed for the random number generator.
    """

    def __init__(self,
                 dataset_reader: DatasetReader,
                 model: str,
                 task: str,
                 sentence_transformer: str = '',
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.pipeline = pipeline(task=task, model=model)
        self.train_ds = dataset_reader.dataset[index_split]
        self.test_ds = dataset_reader.dataset[test_split]
        if sentence_transformer == '':
            try:
                self.encoder = SentenceTransformer(model)
            except:
                self.encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        else:
            self.encoder = SentenceTransformer(sentence_transformer)
        if self.encoder.tokenizer.pad_token is None:
            self.encoder.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        


    def retrieve(self):
        embeddings = self.encoder.encode(["\n".join(self.train_ds[idx][inp] for inp in self.dataset_reader.input_columns) for idx in range(len(self.train_ds))])
        dissimilarity_matrix = 1-cosine_similarity(embeddings)
        print({inp:self.train_ds[0][inp] for inp in self.dataset_reader.input_columns})
        confidences = [1-self.pipeline({inp:self.train_ds[idx][inp] for inp in self.dataset_reader.input_columns})['score'] for idx in range(len(self.train_ds))]

        highest_confidence_idxs = sorted(list(range(len(confidences))), key=lambda idx: confidences[idx])[:self.ice_num]
        initial_genotype = np.zeros(len(confidences))
        for idx in highest_confidence_idxs:
            initial_genotype[idx] = 1

        def qkp_objective_function(selections):
            confidence_weight = np.sum(confidences*selections)
            dissimilarity_weight = np.sum(dissimilarity_matrix*np.outer(selections, selections))
            return confidence_weight * dissimilarity_weight
        
        ga = GeneticAlgorithm(population_size=100, set_size=len(embeddings), amount_selections=self.ice_num,
                            fitness_function=FitnessFunction(obj_func=qkp_objective_function, evaluation_budget=1e6),
                            initial_genotype=initial_genotype)

        rtr_idx_list, _ = ga.run()

        return [list(rtr_idx_list)] * len(self.test_ds)





from typing import Callable
import numpy as np

class Individual:

    def __init__(self, set_size: int, amount_selections: int, genotype: np.ndarray = None):
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

    def __init__(self, population_size: int, set_size: int, amount_selections: int, fitness_function: FitnessFunction,
                 initial_genotype: np.ndarray):
        self.set_size = set_size
        self.amount_selections = amount_selections
        self.population_size = population_size

        initialized_individuals = int(0.1*self.population_size)

        self.population = [Individual(set_size, amount_selections, initial_genotype) for _ in range(initialized_individuals)] +\
                        [Individual(set_size, amount_selections) for _ in range(self.population_size - initialized_individuals)]
        
        self.fitness_function = fitness_function

    def run(self):
        while self.fitness_function.evaluations < self.fitness_function.evaluation_budget:
            self.create_offspring()
            sorted_population = sorted(self.population, key=lambda x: -self.fitness_function.evaluate(x))
            self.population = sorted_population[:self.population_size]
        sorted_population = sorted(self.population, key=lambda x: -self.fitness_function.evaluate(x))
        print(sorted_population[0].genotype)
        return np.where(sorted_population[0].genotype == 1)[0], self.fitness_function.evaluate(sorted_population[0])

    def create_offspring(self):
        offspring = []
        for individual in self.population:
            selected_genes = np.where(individual.genotype == 1)[0]
            not_selected_genes = np.where(individual.genotype == 0)[0]
            unselect_gene = np.random.choice(selected_genes)
            select_gene = np.random.choice(not_selected_genes)
            new_genotype = individual.genotype.copy()
            new_genotype[unselect_gene] = 0
            new_genotype[select_gene] = 1
            offspring.append(Individual(set_size=self.set_size, amount_selections=self.amount_selections, genotype=new_genotype))
        self.population += offspring
