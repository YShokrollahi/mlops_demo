# scripts/app.py
from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)
model = mlflow.sklearn.load_model("models/model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
