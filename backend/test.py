from ICLModel import ICLModel

model = ICLModel(
    {
        "api": False,
        "model": "roberta-large",
        "engine": None,
        "ppl_support": True
    },
    "GenInferencer",
    "tasksource/bigbench",
    10,
    "TopkRetriever",
    3,
	)

print(model.run())