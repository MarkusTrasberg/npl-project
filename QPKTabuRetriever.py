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


class QPKTabuRetriever(BaseRetriever):
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
				 sentence_transformer: str = 'sentence-transformers/all-mpnet-base-v2',
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

		if task == 'question-answering':
			self.confidence_columns = ["question", "context"]
		elif task == 'sentiment-analysis':
			self.confidence_columns = ["text"]
		if 'flan' in model:
			self.pipeline = pipeline(task=task, model="gpt2")
		else:
			try:
				self.pipeline = pipeline(task=task, model=model)
			except:
				self.pipeline = pipeline(task=task, model="gpt2-large")
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
		embeddings = self.encoder.encode(["\n".join(self.train_ds[idx][inp] for inp in self.confidence_columns) for idx in range(len(self.train_ds))])
		dissimilarity_matrix = 1-cosine_similarity(embeddings)
		confidences = [self.pipeline({inp:self.train_ds[idx][inp] for inp in self.confidence_columns})['score'] for idx in range(len(self.train_ds))]

		ts = TabuSearch(5, int(1e4), confidences, dissimilarity_matrix)

		highest_confidence_idxs = sorted(list(range(len(confidences))), key=lambda idx: confidences[idx])[:self.ice_num]
		initial_genotype = np.zeros(len(confidences))
		for idx in highest_confidence_idxs:
			initial_genotype[idx] = 1

		best_genotype = ts.run(initial_genotype)
		rtr_idx_list = [i for i in range(len(self.train_ds)) if best_genotype[i] == 1]
		to_return = list()
		for _ in range(len(self.test_ds)):
			np.random.shuffle(rtr_idx_list)
			to_return.append(list(rtr_idx_list))
		return to_return

class TabuSearch:
	 
	def __init__(self, tabu_size: int, iterations: int, confidences: np.ndarray, dissimilarity_matrix: np.ndarray):
		self.tabu_size = tabu_size
		self.iterations = iterations
		self.confidences = confidences
		self.dissimilarity_matrix = dissimilarity_matrix

	def run(self, initial_solution: np.ndarray):
		"""Running the tabu search."""

		best_solution = initial_solution
		best_candidate = initial_solution
		best_eval = self.qkp_objective_function(best_candidate)
		tabu_list = [initial_solution]

		eval_map = {self.candidate_to_string(initial_solution):self.qkp_objective_function(initial_solution)}

		for i in range(self.iterations):

			if i % 100 == 0:
				print(f"iteration: {i}")

			best_candidate, best_candidate_eval, solution_neighboorhood = self.get_neighboorhood_solutions(best_candidate, tabu_list)
			eval_map.update(solution_neighboorhood)

			if best_candidate_eval > best_eval:
				best_solution = best_candidate

			tabu_list.append(best_candidate)
			if len(tabu_list) > self.tabu_size:
				removed_candidate = tabu_list.pop(0)
				if self.candidate_to_string(removed_candidate) in eval_map.keys():
					del eval_map[self.candidate_to_string(removed_candidate)]			

		return best_solution


	def get_neighboorhood_solutions(self, candidate: np.ndarray, tabu_list: list):
		new_candidates = self.create_candidates(candidate, tabu_list)
		new_evals = [self.qkp_objective_function(new_c) for new_c in new_candidates]

		sorted_candidate_idxs = sorted(list(range(len(new_candidates))), key=lambda idx: -new_evals[idx])
		sorted_evals = [new_evals[idx] for idx in sorted_candidate_idxs]
		sorted_candidates = [new_candidates[idx] for idx in sorted_candidate_idxs]

		return sorted_candidates[0], sorted_evals[0], {self.candidate_to_string(sorted_candidates[idx]):sorted_evals[idx] for idx in range(len(sorted_evals))}

	def candidate_to_string(self, candidate: np.ndarray):
		return "".join(str(x) for x in candidate)

	def create_candidates(self, candidate: np.ndarray, tabu_list: list):
		candidates = list()
		tabu_list_str = [self.candidate_to_string(t) for t in tabu_list]
		for _ in range(self.tabu_size):
			counter = 5
			new_candidate = candidate.copy()
			while counter > 0 and self.candidate_to_string(new_candidate) in tabu_list_str:
				new_candidate = self.mutate(new_candidate)
				counter -= 1
			candidates.append(new_candidate)
		return candidates

	def mutate(self, candidate: np.ndarray):
		selected_genes = np.where(candidate == 1)[0]
		not_selected_genes = np.where(candidate == 0)[0]
		unselect_gene = np.random.choice(selected_genes)
		select_gene = np.random.choice(not_selected_genes)
		new_genotype = candidate.copy()
		new_genotype[unselect_gene] = 0
		new_genotype[select_gene] = 1
		return new_genotype

	def qkp_objective_function(self, selections):
				confidence_weight = np.sum(self.confidences*selections)
				dissimilarity_weight = np.sum(self.dissimilarity_matrix*np.outer(selections, selections))
				return confidence_weight * dissimilarity_weight
