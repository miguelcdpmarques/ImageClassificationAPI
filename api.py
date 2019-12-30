from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return jsonify({'message': 'Hello World'})


@app.route('/model', methods=['POST'])
def model():
    input_img = request.files['photo'] 
    return jsonify({'fruitName': 'Banana', 'predictionConfidence': 0.1})


if __name__ == "__main__":
    app.run(debug=True)