from app import create_app
from flask import Flask, request, jsonify

# Create an application instance
app = create_app()

# Define a route to fetch the available articles



@app.route("/learn", methods=["GET", "POST"], strict_slashes=False)
def articles():
    if request.method == 'POST':
        response = jsonify({'Got this json':request.data.decode('utf-8')})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response



if __name__ == "__main__":
    app.run(debug=True)