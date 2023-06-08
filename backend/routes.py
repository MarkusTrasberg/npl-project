from app import create_app
from flask import Flask, request, jsonify

# Create an application instance
app = create_app()

# Define a route to fetch the available articles

MODELS = ["gpt3", "opt-175b", "gpt2", "flan-t5-small"] # Todo extend list
MODEL_ENGINES = {
    "gpt3": ["text-davinci-003"],  # Todo extend list
}
DATASETS = ["gpt3mix/sst2"] # Todo extend list
INFERENCERS = ["PPLInferencer", "GenInferencer", "CoTInferencer"]
RETRIEVERS = ["RandomRetriever", "BM25Retriever", "TopkRetriever",
               "VotekRetriever", "DPPRetriever", "MDLRetriever", "ZeroRetriever"]



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
                models:
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
    responses:
      200:
        description: Successful response
    """
    parameters = {
        "models": MODELS,
        "model_engines": MODEL_ENGINES,
        "inferencers": INFERENCERS,
        "datasets": DATASETS,
        "retrievers": RETRIEVERS
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
                model:
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
                retriever:
                    type: string
                    example: TopkRetriever
                ice_size:
                    type: integer
                    examples: 3
        - name: output_data
          in: body
          description: ICL run score
          required: True
          schema:
            type: object
            required: true
            properties:
                score:
                    type: number
                    example: 0.99
    responses:
      200:
        description: Successful response
      400:
        description: Bad request
    """
    request_json = request.get_json()



    response = jsonify({'Got this json':request.data.decode('utf-8')})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(debug=True)