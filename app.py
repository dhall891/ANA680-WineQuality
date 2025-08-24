
import pickle
from flask import Flask, request, jsonify, render_template

# Load trained model
with open("wine_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return "Wine Quality Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input features from JSON request
        data = request.get_json(force=True)
        features = [
            data['fixed_acidity'],
            data['volatile_acidity'],
            data['citric_acid'],
            data['residual_sugar'],
            data['chlorides'],
            data['free_sulfur_dioxide'],
            data['total_sulfur_dioxide'],
            data['density'],
            data['pH'],
            data['sulphates'],
            data['alcohol']
        ]
        prediction = model.predict([features])
        return jsonify({'predicted_quality': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
