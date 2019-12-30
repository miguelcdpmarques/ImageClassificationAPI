import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from predict_pipeline import PredictPipeline

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return jsonify({'message': 'Hello World'})


@app.route('/model', methods=['POST'])
def model():
    file = request.files['photo']
    filename = secure_filename(file.filename)
    file.save(os.path.join('uploaded_photos', filename))
    pipeline = PredictPipeline(json_model_path='model.json', model_weights_path='model_weights.h5',img_path='uploaded_photos/' + filename)
    prediction = pipeline.image_class_predict()
    return jsonify({'fruitName': prediction, 'predictionConfidence': 0.1})


if __name__ == "__main__":
    app.run(debug=True)