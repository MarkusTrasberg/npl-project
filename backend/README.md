# Backend of the ICL demo tool

To run this backend you need to have Python installed. 

First navigate to the backend directory by executing the following command

```bash
cd backend
```

## Create .env

Create a .env file in the directory manually or by executing the following command

```'bash
touch .env
```

In this file add the following line (note replace ```<Token>``` with a token created at [https://onboard.openai.com](https://onboard.openai.com/))
```
OPENAI_API_KEY=<Token>
```


## Install dependencies

Setup a virtual environment, if desired. <https://docs.python.org/3/library/venv.html>. 
Execute the following command to install dependencies

```bash
pip install -r requirements.txt
```

## Running the backend

To run the backend execute the following command

```bash
./run
```

This will open a Flask server which is available at <localhost:8000>. To see all available API endpoints navigate to <localhost:8000/apidocs> in the browser

## Accelerate config

This project makes use of the Accelerate library to speed up processing. This exact configuration can be tweaked in the ```accelerate_config.yaml``` file
