import os
from dotenv import load_dotenv
from openicl import *
import openai
import os
from dotenv import load_dotenv
from datasets import load_dataset

class ICLModel():
    
	def __init__(self,
	      	model: str,
			use_api: bool,
			model_engine: str,
			inferencer: str,
			dataset: str,
			dataset_size: int,
			retriever: str,
			ice_size: int,
		):

		self.model = model
		self.use_api = use_api
		self.model_engine = model_engine
		self.inferencer = inferencer
		self.dataset = dataset
		self.dataset_size = dataset_size
		self.retriever = retriever
		self.ice_size = ice_size

		# Get api key when using api
		if self.use_api:
			load_dotenv()
			self.openai.api_key = os.getenv('OPENAI_API_KEY')

		self.setDatasetSpecifics()

		self.setRetriever()

		self.setRetriever()

	def setDatasetReader(self):
		# Todo set dataset specifics based on dataset
		# Todo training/test split

		print("Loading dataset...")

		tp_str = "</E></Q>\t</L>"  
		self.prompt_template = PromptTemplate(tp_str, column_token_map={'question' : '</Q>', 'logical_form' : '</L>'}, ice_token='</E>')

		self.dataset_input_columns = []
		self.dataset_output_column = ""

		ds = load_dataset(self.dataset)
		self.dsr = DatasetReader(ds, ds_size=self.dataset_size, input_columns=self.dataset_input_columns, output_column=self.dataset_output_column)  

	def setRetriever(self):
		# Todo can this be better?

		print("Initiating retriever...")

		if (self.retriever == "RandomRetriever"):
			self.rtvr = RandomRetriever(self.dsr, ice_num=self.ice_size)

		elif (self.retriever == "BM25Retriever"):
			self.rtvr = BM25Retriever(self.dsr, ice_num=self.ice_size)

		elif (self.retriever == "TopkRetriever"):
			self.rtvr = TopkRetriever(self.dsr, ice_num=self.ice_size)

		elif (self.retriever == "VotekRetriever"):
			self.rtvr = VotekRetriever(self.dsr, ice_num=self.ice_size)

		elif (self.retriever == "DPPRetriever"):
			self.rtvr = DPPRetriever(self.dsr, ice_num=self.ice_size)

		elif (self.retriever == "MDLRetriever"):
			self.rtvr = MDLRetriever(self.dsr, ice_num=self.ice_size)

		elif (self.retriever == "ZeroRetriever"):
			self.rtvr = ZeroRetriever(self.dsr, ice_num=self.ice_size)

		else:
			raise Exception("Retriever parameter is not correct")


	def setInferencer(self):
		# Todo can this be better?
		# Todo set inferencer specifics
		# Todo check if every inferencer needs different paramaters

		print("Initiating inferencer...")

		if self.use_api:
			# Todo check if model_engine works like this
			if (self.retriever == "PPLInferencer"):
				self.infr =  PPLInferencer(api_name=self.model, engine=self.model_engine, sleep_time=3)

			elif (self.retriever == "GenInferencer"):
				self.infr = GenInferencer(api_name=self.model, engine=self.model_engine, sleep_time=3)

			elif (self.retriever == "CoTInferencer"):
				self.infr = CoTInferencer(api_name=self.model, engine=self.model_engine, sleep_time=3)

			else:
				raise Exception("Inferencer parameter is not correct")
		else:
			if (self.retriever == "PPLInferencer"):
				self.infr = PPLInferencer(model=self.model)

			elif (self.retriever == "GenInferencer"):
				self.infr = GenInferencer(model=self.model)

			elif (self.retriever == "CoTInferencer"):
				self.infr = CoTInferencer(model=self.model)

			else:
				raise Exception("Inferencer parameter is not correct")

	
	def run(self):
		print("Running and calculating score...")

		return self.infr.inference(self.rtvr, ice_template=self.prompt_template)



