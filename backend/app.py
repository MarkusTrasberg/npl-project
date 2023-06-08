# Import the required packages
from flask import Flask
from flasgger import Swagger


def create_app():
    """Application-factory pattern"""
    app = Flask(__name__)
    swagger = Swagger(app)

    return app