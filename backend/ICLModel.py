import json
import os
from dotenv import load_dotenv
from openicl import DatasetReader, PromptTemplate, AccEvaluator, BleuEvaluator
import openai
import os
from dotenv import load_dotenv
from datasets import load_dataset
import importlib
from accelerate import Accelerator
import numpy as np


# Todo extend with the new dataset and see if there is a train/test division
DATASET_SPECIFICS = {
	"gpt3mix/sst2": {
		"name": None,
		"input_output": (['text'], 'label'),
		"rename_columns": None,
		"prompt_template_gen": PromptTemplate(
			template="</E>Review:</R>\nSentiment (0=positive, 1=negative):<\L>",
			column_token_map={'text': '</R>', 'label': '<\L>'}, 
			ice_token='</E>'
		),
		"prompt_template_ppl": PromptTemplate(
			template={
				0: '</E>Positive Movie Review: </R>',
				1: '</E>Negative Movie Review: </R>' 
			},
			column_token_map={'text': '</R>'},
			ice_token='</E>'
		),
		"score_function": "score_sentiment",
		"preprocess": None,
	},
	"imdb": {
		"name": None,
		"input_output": (["text"], 'label'),
		"rename_columns": None,
		"prompt_template_gen": PromptTemplate(
			template="</E>Review:</R>\nSentiment (0=positive, 1=negative):<\L>",
			column_token_map={'text': '</R>', 'label': '<\L>'}, 
			ice_token='</E>'
		),
		"prompt_template_ppl": PromptTemplate(
			template={
				0: '</E>Positive Movie Review: </R>',
				1: '</E>Negative Movie Review: </R>' 
			},
			column_token_map={'text': '</R>'},
			ice_token='</E>'
		),
		"score_function": "score_sentiment",
		"preprocess": None,
	},
	"tasksource/bigbench": {
		"name": "disambiguation_qa",
		"input_output": (["question", "A", "B", "C"], 'answer'),
		"rename_columns": [("inputs", "question"), ('multiple_choice_scores',"answer")],
		"prompt_template_gen": PromptTemplate(
			template="</E>Multiple choice question:\n</Q>\nChoices:\nA:</Ans1>\nB:</Ans2>\nC:</Ans3>\nAnswer:</TrueAns>",
			column_token_map={'question':'</Q>', 'A': '</Ans1>', 'B': '</Ans2>', 'C': '</Ans3>', "answer": "</TrueAns>"},
        	ice_token='</E>'
		),
		"promt_template_ppl": PromptTemplate(
			template={
				'A': "</E>Answer to multiple choice question:</Q>",
				'B': "</E>Answer to multiple choice question:</Q>",
				'C': "</E>Answer to multiple choice question:</Q>",
			},
			column_token_map={'question':'</Q>', 'A': '</Ans1>', 'B': '</Ans2>', 'C': '</Ans3>'},
			ice_token='</E>' 
		),
		"score_function": "score_mc_question",
		"preprocess": "bb_pre_process",
	},
	"commonsense_qa": {
		"name": None,
		"input_output": (["question", "context", "A", "B", "C", "D", "E"], 'answer'),
		"rename_columns": [("question_concept", "context"), ('answerKey',"answer")],
		"prompt_template_gen": PromptTemplate(
			template="</E>Multiple choice question:\n</Q>\nChoices:\nA:</Ans1>\nB:</Ans2>\nC:</Ans3>\nD:</Ans4>\nE:</Ans5>\nAnswer:</TrueAns>",
			column_token_map={'question':'</Q>', 'A': '</Ans1>', 'B': '</Ans2>', 'C': '</Ans3>', 'D': '</Ans4>', 'E': '</Ans5>', "answer": "</TrueAns>"},
        	ice_token='</E>'
		),
		"promt_template_ppl": PromptTemplate(
			template={
				'A': "</E>Answer to multiple choice question:</Q>",
				'B': "</E>Answer to multiple choice question:</Q>",
				'C': "</E>Answer to multiple choice question:</Q>",
				'D': "</E>Answer to multiple choice question:</Q>",
				'E': "</E>Answer to multiple choice question:</Q>",
			},
			column_token_map={'question':'</Q>', 'A': '</Ans1>', 'B': '</Ans2>', 'C': '</Ans3>', 'D': '</Ans4>', 'E': '</Ans5>'},
			ice_token='</E>' 
		),
		"score_function": "score_mc_question",
		"preprocess": "cmqa_pre_process"
	},
	"wmt16": {
		"name": "de-en",
		"input_output": (["de"], 'en'),
		"rename_columns": None,
		"prompt_template": PromptTemplate(
			template='</E></de> = </en>',
			column_token_map={'en': '</en>', 'de': '</de>'}, 
			ice_token='</E>'
		),
		"score_function": "score_translation",
		"preprocess": None,
	}
}

def cmqa_pre_process(example):
    for i in range(5):
        example[chr(ord('A') + i)] = example['choices']['text'][i]
    return example

def bb_pre_process(example):
    for i in range(3):
        example[chr(ord('A') + i)] = example['multiple_choice_targets'][i]
    example['multiple_choice_scores'] = chr(ord('A') + np.where(np.array(example['multiple_choice_scores']) == 1)[0][0])
    example['context'] = "Disambiguation"
    return example

class ICLModel():
    
	def __init__(self,
	      	model_dict: dict,
			inferencer: str,
			dataset: str,
			dataset_size: int,
			retriever: str,
			ice_size: int,
		):

		self.model_dict = model_dict
		self.inferencer = inferencer
		self.dataset = dataset
		self.dataset_size = dataset_size
		self.retriever = retriever
		self.ice_size = ice_size
		self.use_api = model_dict['api']
		self.prompt_template = None
		self.dsr = None
		self.rtvr = None
		self.infr = None
		self.score_function = None
	
		self.setDatasetReader()

		self.setRetriever()

		self.setInferencer()

	def setDatasetReader(self):
		print("Loading dataset...")

		# Get dataset specifics
		dataset_specifics = DATASET_SPECIFICS[self.dataset]

		dataset_input_columns, dataset_output_column = dataset_specifics["input_output"]
		self.score_function = dataset_specifics["score_function"]

		# Todo change, not future proof
		if self.infr == "PPLInferencer":
			self.prompt_template = dataset_specifics["prompt_template_ppl"]
		else:
			self.prompt_template = dataset_specifics["prompt_template_gen"]

		# Choose a subset of a dataset
		if dataset_specifics['name'] is not None:
			ds = load_dataset(self.dataset, split="train", name=dataset_specifics['name'])
		else:
			ds = load_dataset(self.dataset, split="train")

		# Set train and test split
		ds = ds.train_test_split(test_size=self.dataset_size, train_size=self.dataset_size, shuffle=True)

		# Preprocess
		if dataset_specifics['preprocess'] is not None:
			ds = eval(f"ds.map({dataset_specifics['preprocess']})")

		# Rename columns
		if dataset_specifics['rename_columns'] is not None:
			for old_name, new_name in dataset_specifics['rename_columns']:
				ds = ds.rename_column(old_name, new_name)	
	
		self.dsr = DatasetReader(ds, input_columns=dataset_input_columns, output_column=dataset_output_column)

	def setRetriever(self):
		# Todo set retriever specifics
		# Todo check if each retriever needs different paramaters
			# Todo randomrtvr needs seed
			# Todo bm25tvr needs index_corpus, test_corpus and bm25
			# Todo topkrtvr needs batch_size, model, tokenizer and index
			# Todo votekrtvr needs batch_size, model, tokenizer, index and votek_k
			# Todo dpprtvr needs batch_size, model, tokenizer, index, seed and scale_factor
			# Todo mdlrtvr needs batch_size, model, tokenizer, index, select_time, labels and seed
			# Todo zerortvr

		print("Initiating retriever...")

		try:
			RetrieverClass = getattr(importlib.import_module("openicl"), self.retriever)

			# Accelerator specifics are defined by 'accelerator_config.yaml'
			self.rtvr = RetrieverClass(self.dsr, ice_num=self.ice_size, accelerator=Accelerator())
		except:
			raise Exception("Retriever parameters are not correct")


	def setInferencer(self):
		# Todo set inferencer specifics
		# Todo check if each inferencer needs different paramaters
			# Todo pplinfr needs labels
			# Todo geninfr needs gen_field_replace_token and generation_kwargs
			# Todo cotinfr needs gen_field_replace_token, generation_kwargs and cot_list

		print("Initiating inferencer...")

		try:
			InferencerClass = getattr(importlib.import_module("openicl"), self.inferencer )
			if self.use_api:
				# Get api key
				load_dotenv()
				openai.api_key = os.getenv('OPENAI_API_KEY')

				# Set inferencer
				self.infr = InferencerClass(api_name=self.model_dict['model'], engine=self.model_dict['engine'], sleep_time=3)
			else:
				self.infr = InferencerClass(model=self.model_dict['model'])
		except:
			raise Exception("Inferencer parameter is not correct")

	
	def run(self):
		print("Running and calculating score...")
		
		predictions = self.infr.inference(self.rtvr, ice_template=self.prompt_template)
		
		with open("icl_inference_output/predictions.json") as f:
			data = json.load(f)

		score = eval(f"self.{self.score_function}(predictions)")

		result = {
			"accuracy": score,
			"origin_prompt": [value["origin_prompt"] for value in data.values()],
			"predictions": predictions,
			"answers": self.dsr.references
		}
		return result


	def score_sentiment(self, predictions):
		evaluator = AccEvaluator()

		predictions = [int(p) for p in predictions]
		
		score = evaluator.score(predictions=predictions, references=self.dsr.references)

		return score['accuracy']

	def score_mc_question(self, predictions):
		evaluator = AccEvaluator()

		score = evaluator.score(predictions=predictions, references=self.dsr.references)

		return score['accuracy']

	def score_translation(self, predictions):
		evaluator = BleuEvaluator()

		score = evaluator.score(predictions=predictions, references=self.dsr.references)

		return score['sacrebleu']
