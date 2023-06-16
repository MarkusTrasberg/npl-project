from ICLModel import ICLModel

model = ICLModel(
    {
        "api": True,
        "model": "gpt3",
        "engine": "text-davinci-003",
        "ppl_support": False
    },
    "GenInferencer",
    "imdb",
    10,
    "TopkRetriever",
    1,
	)

print(model.run())

print({
        'accuracy': 1.0,
        'questions': ["What is the Answer to the Ultimate Question of Life, The Universe, and Everything?"],
        'predictions': ['42'],
        'answers': ['42 '],
})