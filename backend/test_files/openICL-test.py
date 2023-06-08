print("Importing packages...")
from openicl import DatasetReader, PromptTemplate, BM25Retriever, GenInferencer
import openai
import os
from dotenv import load_dotenv
from datasets import load_dataset
import nltk
nltk.download("punkt")

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load dataset
print("Loading dataset...")
dataset = load_dataset("iohadrubin/mtop")
dataset['train'] = dataset['train'].select([0, 1, 2])
dataset['test'] = dataset['test'].select([0])

dr = DatasetReader(dataset, input_columns=['question'], output_column='logical_form')  

tp_str = "</E></Q>\t</L>"      
tp = PromptTemplate(tp_str, column_token_map={'question' : '</Q>', 'logical_form' : '</L>'}, ice_token='</E>')

print("Initiating retriever...")
rtr = BM25Retriever(dr, ice_num=1)

print("Initiating inferencer...")
infr = GenInferencer(api_name='gpt3', engine='text-davinci-003', sleep_time=3)

print("Running and calculating score...")
print(infr.inference(rtr, ice_template=tp))








