import os
from app import create_app
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import json

from ICLModel import ICLModel

# Create an application instance
app = create_app()
CORS(app)
# app.config.from_file("flask_config.json", load=json.load)

# Todo GPT3 doesnt support PPL
INFERENCERS = ["PPLInferencer", "GenInferencer", "CoTInferencer"]
RETRIEVERS = ["RandomRetriever", "BM25Retriever", "TopkRetriever",
                "VotekRetriever", "DPPRetriever", "MDLRetriever", "ZeroRetriever"]
EVALUATORS = ["AccEvaluator", "BleuEvaluator", "RougeEvaluator", "SquadEvaluator"]
# Todo extend list
MODELS = {
    "gpt3/text-davinci-003": {
        "api": True,
        "model": "gpt3",
        "engine": "text-davinci-003"
    }, 
    "gpt3/ada": {
        "api": True,
        "model": "gpt3",
        "engine": "ada"
    }, 
    "gpt3/babbage": {
        "api": True,
        "model": "gpt3",
        "engine": "babbage"
    }, 
    "gpt3/curie": {
        "api": True,
        "model": "gpt3",
        "engine": "curie"
    }, 
    "gpt3/davinci": {
        "api": True,
        "model": "gpt3",
        "engine": "davinci"
    }, 
    "google/flan-t5-small": {
        "api": False,
        "model": "flan-t5-small",
        "engine": None
    }, 
    "gpt2": {
        "api": False,
        "model": "gpt2",
        "engine": None
    }, 
}
DATASETS = ["gpt3mix/sst2", "iohadrubin/mtop"] # Todo extend list and see if there is a train/test division


# Define a route to fetch the available parameters
@app.route("/parameters", methods=["GET"], strict_slashes=False)
def parameters():
    """
    Do an In-Context Learning run based on the specified paramters
    ---
    consumes:
      - application/json
    parameters:
        - name: output_data
          in: body
          description: All possible ICL parameters
          required: True
            schema:
                type: object
                required: True
                properties:
                    models:
                        type: object
                        description: |
                            A dictionary of models that can be selected.
                    inferencers:
                        type: array
                        items:
                            type: string
                        description: |
                            An array of inferencers that can be selected.
                    datasets:
                        type: array
                        items:
                            type: string
                        description: |
                            An array of datasets that can be selected.
                    retrievers:
                        type: array
                        items:
                            type: string
                        description: |
                            An array of retrievers that can be selected.
                    evaluators:
                        type: array
                        items:
                            type: string
                        description: |
                            An array of evaluators that can be selected.
    responses:
      200:
        description: Successful response
    """
    parameters = {
        "models": list(MODELS.keys()),
        "inferencers": INFERENCERS,
        "datasets": DATASETS,
        "retrievers": RETRIEVERS,
        "evaluators": EVALUATORS,
    }
    response = jsonify(parameters)
    return response

@app.route("/debug", methods=["POST"], strict_slashes=False)
def debug():
    """
    Do an In-Context Learning debug run without openicl based on the specified parameters
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: ICL parameters values
          required: True
          schema:
            type: object
            required: True
            properties:
                model:
                    type: string
                    example: gpt3/ada
                    description: |
                        Dictionary key value to select model
                inferencer:
                    type: string
                    example: GenInferencer
                    description: |
                        Selected inferencer
                dataset:
                    type: string
                    example: gpt3mix/sst2
                    description: |
                        Selected dataset
                dataset_size:
                    type: integer
                    example: 100
                    description: |
                        Selected dataset size
                retriever:
                    type: string
                    example: TopkRetriever
                    description: |
                        Selected retriever
                ice_size:
                    type: integer
                    examples: 3
                    description: |
                        Selected in-context example size
                evaluator:
                    type: string
                    example: AccEvaluator
                    description: |
                        Selected evaluator
        - name: output_data
          in: body
          description: ICL run values
          required: True
          schema:
            type: object
            required: true
            properties:
                accuracy:
                    type: number
                    example: 0.99
                    description: |
                        Accuracy score returned by the evaluator
                origin_prompt:
                    type: array
                    items:
                        type: string
                    description: |
                        An array of the original prompts
                predictions:
                    type: array
                    items:
                        type: string
                    description: |
                        An array of predicted answers to the test questions
                answers:
                    type: array
                    items:
                        type: string
                    description: |
                        An array of true answers to the tested questions
                
    responses:
      200:
        description: Successful response
      400:
        description: Bad request
    """

    request_json = request.get_json()
    print(request_json)

    response = jsonify({
        'accuracy': 1.0,
        'questions': ["What is the Answer to the Ultimate Question of Life, The Universe, and Everything?"],
        'predictions': ['42'],
        'answers': ['42 '],
        })

    # model_name = request_json["model_name"]
    # inferencer = request_json["inferencer"]
    # datasets = request_json["datasets"]
    # dataset_size = request_json["dataset_size"]
    # retriever = request_json["retriever"]
    # ice_size = request_json["ice_size"]
    # evaluator = request_json["evaluator"]
    
    return response

@app.route("/run", methods=["POST"], strict_slashes=False)
def run():
    """
    Do an In-Context Learning run through openicl, based on the specified parameters
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: ICL parameters values
          required: True
          schema:
            type: object
            required: True
            properties:
                model:
                    type: string
                    example: gpt3/ada
                    description: |
                        Dictionary key value to select model
                inferencer:
                    type: string
                    example: GenInferencer
                    description: |
                        Selected inferencer
                dataset:
                    type: string
                    example: gpt3mix/sst2
                    description: |
                        Selected dataset
                dataset_size:
                    type: integer
                    example: 100
                    description: |
                        Selected dataset size
                retriever:
                    type: string
                    example: TopkRetriever
                    description: |
                        Selected retriever
                ice_size:
                    type: integer
                    examples: 3
                    description: |
                        Selected in-context example size
                evaluator:
                    type: string
                    example: AccEvaluator
                    description: |
                        Selected evaluator
        - name: output_data
          in: body
          description: ICL run values
          required: True
          schema:
            type: object
            required: true
            properties:
                accuracy:
                    type: number
                    example: 0.99
                    description: |
                        Accuracy score returned by the evaluator
                origin_prompt:
                    type: array
                    items:
                        type: string
                    description: |
                        An array of the original prompts
                predictions:
                    type: array
                    items:
                        type: string
                    description: |
                        An array of predicted answers to the test questions
                answers:
                    type: array
                    items:
                        type: string
                    description: |
                        An array of true answers to the tested questions
                
    responses:
      200:
        description: Successful response
      400:
        description: Bad request
    """
    request_json = request.get_json()

    model_key = request_json["model"]
    model = MODELS[model_key]
    inferencer = request_json["inferencer"]
    datasets = request_json["datasets"]
    dataset_size = request_json["dataset_size"]
    retriever = request_json["retriever"]
    ice_size = request_json["ice_size"]
    evaluator = request_json["evaluator"]

    if ice_size > dataset_size:
        return Response(
                "In-Context example size cant be larger than dataset size",
                status=400,
            )

    for dataset in datasets:
        try:
            model = ICLModel(model, inferencer, dataset, dataset_size, retriever, ice_size, evaluator)
            result = model.run()
        except Exception as e:
            return Response(
                str(e),
                status=400,
            )
    
    # TODO: Multiple responses so that we can let the frontend know at what stage of the run we are (i.e. inferencing, predicting)
    # TODO: Currently only returns results[0] but should combine answers.

    response = jsonify(result)
    return response


if __name__ == "__main__":
    app.run(host=os.getenv('IP', '127.0.0.1'), debug=True, port=int(os.getenv('PORT', 8000)))