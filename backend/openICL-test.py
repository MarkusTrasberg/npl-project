print("Importing packages...")
from openicl import DatasetReader, PromptTemplate, CoTInferencer, PPLInferencer,BM25Retriever, VoteKRetriever, TopkRetriever,  RandomRetriever, DPPRetriever, MDLRetriever, ZeroRetriever
import openai
import os
from dotenv import load_dotenv
from openicl import AccEvaluator

# parameters that need to be retrieved from the frontend
model_name = 'distilgpt2'
ice_num = 4
retriever_name = "topK"
inferencer_name = "ppl"


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load dataset
print("Loading dataset...")
data = DatasetReader('gsm8k', name='main',
						input_columns=['question'], output_column='answer', ds_size=100)

template = PromptTemplate('</E> Question: </Q> \n Answer: </A>',
							{'question':'</Q>', 'answer':'</A>'},
							ice_token='</E>')

print("Initiating retriever and inferencer...")

# Initiate the retriever (heuristic)
# topK, bm25, random, voteK, dpp, mdl, zero 

if retriever_name == "topK":
    retriever = TopkRetriever(data, ice_num=ice_num)
elif retriever_name ==	"bm25":
	retriever = BM25Retriever(data, ice_num=ice_num)
elif retriever_name == "random":
      retriever = RandomRetriever(data, ice_num=ice_num)
elif retriever_name == "voteK":
      retriever = VoteKRetriever(data, ice_num=ice_num)
elif retriever_name == "dpp":
      retriever = DPPRetriever(data, ice_num=ice_num)
elif retriever_name == "mdl":
      retriever = MDLRetriever(data, ice_num=ice_num)
else:
      retriever = ZeroRetriever(data)
      

# Inference by Chain-of-Thought
cot_list=["Let's think step by step.",
			"\nTherefore, the answer (arabic numerals) is"]

# inferencer = CoTInferencer(cot_list=cot_list, api_name='ada')
inferencer = PPLInferencer(model_name=model_name)

print("Running and calculating score...")
predictions = inferencer.inference(retriever, ice_template=template)

print(predictions)

# compute accuracy for the prediction
score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)
# data = DatasetReader('gpt3mix/sst2', input_columns=['text'], output_column='label')

# # Define the prompt template for the task
# tp_dict = {
#     0: '</E> Positive Movie Review: </Q>',
# 	1: '</E> Negative Movie Review: </Q>'
# }

# template = PromptTemplate(tp_dict, {'text':'</Q>'}, ice_token='</E>')


# # Initiate the retriever and inferencer
# retriever = TopkRetriever(data, ice_num=8)
# inferencer = PPLInferencer(model_name='gpt2-xl')

# # Run inference and calculate score
# predictions = inferencer.inference(retriever, ice_template=template)
# score = AccEvaluator().score(predictions=predictions, references=data.references)

# print(f'score: ${score}')
