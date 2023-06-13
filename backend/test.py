from ICLModel import ICLModel

model = ICLModel(
    {
        "api": True,
        "model": "gpt3",
        "engine": "text-davinci-003"
    }, 
    "CoTInferencer",
    "gpt3mix/sst2",
    10,
    "TopkRetriever",
    1,
    "AccEvaluator",	
	)

print(model.run())