from app import create_app
from flask import Flask, Response, request, jsonify

from backend.ICLModel import API_NAMES, DATASETS, EVALUATORS, INFERENCERS, MODEL_ENGINES, MODEL_NAMES, RETRIEVERS, ICLModel

# Create an application instance
app = create_app()

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
    response.headers.add('Access-Control-Allow-Origin', '*')
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
                    example: google/flan-t5-small
                api_name:
                    type: string
                    example gpt3
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
    dataset = request_json["dataset"]
    dataset_size = request_json["dataset_size"]
    dataset_split = request_json["dataset_split"]
    retriever = request_json["retriever"]
    ice_size = request_json["ice_size"]
    evaluator = request_json["evaluator"]

    try:
        model = ICLModel(model_name, api_name, model_engine, inferencer, dataset, dataset_size, dataset_split, retriever, ice_size, evaluator)
        result = model.run()
    except Exception as e:
        return Response(
            str(e),
            status=400,
        )
    
    # Todo multiple responses so that we can let the frontend know at what stage of the run we are (i.e. inferencing, predicting)
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(debug=True)