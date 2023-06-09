import os
from dotenv import load_dotenv
from openicl import DatasetReader, PromptTemplate
import openai
import os
from dotenv import load_dotenv
from datasets import load_dataset
import importlib


DATASET_PROMPTEMPLATES = {
	"gpt3mix/sst2": PromptTemplate(
		template={ 
			0: "</E>Positive Movie Review: </text>", 
			1: "</E>Negative Movie Review: </text>"
		}, 
		column_token_map={'text': '</text>'}, 
		ice_token='</E>'
	),
	"iohadrubin/mtop": PromptTemplate(
		template="</E></Q>\t</L>",
		column_token_map={'question' : '</Q>', 'logical_form' : '</L>'}, 
		ice_token='</E>'
	)
}

class ICLModel():
    
	def __init__(self,
	      	model_name: str,
			api_name: bool,
			model_engine: str,
			inferencer: str,
			dataset: str,
			dataset_size: int,
			dataset_split: float,
			retriever: str,
			ice_size: int,
			evaluator: str,
		):

		self.model_name = model_name
		self.api_name = api_name
		self.model_engine = model_engine
		self.inferencer = inferencer
		self.dataset = dataset
		self.dataset_size = dataset_size
		self.dataset_split = dataset_split
		self.retriever = retriever
		self.ice_size = ice_size
		self.evaluator = evaluator
		self.use_api = False
		self.prompt_template = None
		self.dsr = None
		self.rtvr = None
		self.infr = None

		# Get api key when using api
		if self.api_name is not None:
			load_dotenv()
			openai.api_key = os.getenv('OPENAI_API_KEY')
			self.use_api = True

		self.setDatasetReader()

		self.setRetriever()

		self.setInferencer()

	def setDatasetReader(self):
		# Todo training/test split

		print("Loading dataset...")

		self.prompt_template = DATASET_PROMPTEMPLATES[self.dataset]

		dataset_input_columns, dataset_output_column = DATASET_INPUT_OUTPUT[self.dataset]

		ds = load_dataset(self.dataset)
		self.dsr = DatasetReader(ds, ds_size=self.dataset_size, input_columns=dataset_input_columns, output_column=dataset_output_column)  

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
			self.rtvr = RetrieverClass(self.dsr, ice_num=self.ice_size)
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
				self.infr = InferencerClass(api_name=self.api_name, engine=self.model_engine, sleep_time=3)
			else:
				self.infr = InferencerClass(model=self.model_name)
		except:
			raise Exception("Inferencer parameter is not correct")

	
	def run(self):
		print("Running and calculating score...")
		
		predictions = self.infr.inference(self.rtvr, ice_template=self.prompt_template)
		
		try:
			EvaluatorClass = getattr(importlib.import_module("openicl"), self.evaluator)
			evaluator = EvaluatorClass()
		except:
			raise Exception("Evaluator parameter is not correct")
		
		score = evaluator.score(predictions=predictions, references=self.dsr.references)
		return score



