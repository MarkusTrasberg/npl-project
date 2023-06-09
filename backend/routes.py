from app import create_app
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from ICLModel import ICLModel

# Create an application instance
app = create_app()
CORS(app)

DATASET_INPUT_OUTPUT = {
	"gpt3mix/sst2": (['text'], 'label'),
	"iohadrubin/mtop": (['question'], 'logical_form')
}
INFERENCERS = ["PPLInferencer", "GenInferencer", "CoTInferencer"]
RETRIEVERS = ["RandomRetriever", "BM25Retriever", "TopkRetriever",
               			"VotekRetriever", "DPPRetriever", "MDLRetriever", "ZeroRetriever"]
EVALUATORS = ["AccEvaluator", "BleuEvaluator", "RougeEvaluator", "SquadEvaluator"]
MODEL_NAMES = ["gpt2", "google/flan-t5-small"] # Todo extend list
API_NAMES = ["gpt3"]
MODEL_ENGINES = {
    "gpt3": ["text-davinci-003", "ada", "babbage", "curie", "davinci"],  # Todo extend list
}
DATASETS = ["gpt3mix/sst2", "iohadrubin/mtop"] # Todo extend list

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
          description: ICL parameters
          required: True
          schema:
            type: object
            required: true
            properties:
                model_names:
                    type: array
                    example: [gpt2]
                api_names:
                    type: array
                    example: [gpt3]
                model_engines:
                    type: array
                    example: [text-davinci-003]
                inferencers:
                    type: array
                    example: [GenInferencer]
                datasets:
                    type: array
                    example: [gpt3mix/sst2]
                retrievers:
                    type: array
                    example: [TopkRetriever]
                retrievers:
                    type: array
                    example: [AccEvaluator]
    responses:
      200:
        description: Successful response
    """
    parameters = {
        "model_names": MODEL_NAMES,
        "api_names": API_NAMES,
        "model_engines": MODEL_ENGINES,
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
    Do an In-Context Learning debug run without openicl based on the specified paramters
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: ICL parameters to be set
          required: True
          schema:
            type: object
            properties:
                model_name:
                    type: string
                    example: google/flan-t5-small
                api_name:
                    type: string
                    example: gpt3
                model_engine:
                    type: string
                    example: text-davinci-003
                inferencer:
                    type: string
                    example: GenInferencer
                dataset:
                    type: string
                    example: gpt3mix/sst2
                dataset_size:
                    type: integer
                    example: 100
                dataset_split:
                    type: number
                    example: 0.8
                retriever:
                    type: string
                    example: TopkRetriever
                ice_size:
                    type: integer
                    examples: 3
                evaluator:
                    type: string
                    example: AccEvaluator
        - name: output_data
          in: body
          description: ICL run score
          required: True
          schema:
            type: object
            required: true
            properties:
                accuracy:
                    type: number
                    example: 0.99
    responses:
      200:
        description: Successful response
      400:
        description: Bad request
    """

    request_json = request.get_json()
    print(request_json)

    response = jsonify({'accuracy': 0.99})

    model_name = request_json["model_name"]
    inferencer = request_json["inferencer"]
    datasets = request_json["datasets"]
    dataset_size = request_json["dataset_size"]
    dataset_split = request_json["dataset_split"]
    retriever = request_json["retriever"]
    ice_size = request_json["ice_size"]
    evaluator = request_json["evaluator"]
    response = jsonify({'accuracy': 0.99})
    
    return response

@app.route("/run", methods=["POST"], strict_slashes=False)
def run():
    """
    Do an In-Context Learning run based on the specified paramters
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: ICL parameters to be set
          required: True
          schema:
            type: object
            properties:
                model_name:
                    type: string
                api_name:
                    type: string
                    example: gpt3
                model_engine:
                    type: string
                    example: text-davinci-003
                inferencer:
                    type: string
                    example: GenInferencer
                dataset:
                    type: string
                    example: gpt3mix/sst2
                dataset_size:
                    type: integer
                    example: 100
                dataset_split:
                    type: number
                    example: 0.8
                retriever:
                    type: string
                    example: TopkRetriever
                ice_size:
                    type: integer
                    examples: 3
                evaluator:
                    type: string
                    example: AccEvaluator
        - name: output_data
          in: body
          description: ICL run score
          required: True
          schema:
            type: object
            required: true
            properties:
                accuracy:
                    type: number
                    example: 0.99
    responses:
      200:
        description: Successful response
      400:
        description: Bad request
    """
    request_json = request.get_json()

    model_name = request_json["model_name"]
    api_name = request_json["api_name"]
    model_engine = request_json["model_engine"]
    inferencer = request_json["inferencer"]
    datasets = request_json["datasets"]
    dataset_size = request_json["dataset_size"]
    dataset_split = request_json["dataset_split"]
    retriever = request_json["retriever"]
    ice_size = request_json["ice_size"]
    evaluator = request_json["evaluator"]

    results = []
    for dataset in datasets:
        try:
            model = ICLModel(model_name, api_name, model_engine, inferencer, dataset, dataset_size, dataset_split, retriever, ice_size, evaluator)
            result = model.run()
            results.append(result)
        except Exception as e:
            return Response(
                str(e),
                status=400,
            )
    
    # TODO: Multiple responses so that we can let the frontend know at what stage of the run we are (i.e. inferencing, predicting)
    # TODO: Currently only returns results[0] but should combine answers.

    response = jsonify(results[0])
    return response


if __name__ == "__main__":
    app.run(debug=True, port=8000)