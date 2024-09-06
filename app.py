from flask import Flask, request, jsonify, render_template
import numpy as np
import xgboost as xgb
import pickle

# Load your trained model
print("Loading the model...")
model = pickle.load(open('model.pkl', 'rb'))
print("Model loaded successfully.")

app = Flask(__name__)

# Homepage Route
@app.route('/')
def home():
    print("Home page accessed.")
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    print("Prediction route accessed.")
    
    # Get data from form
    input_features = [int(x) for x in request.form.values()]
    print(f"Input features received: {input_features}")
    
    # Prepare input for model
    features = np.array([input_features])
    print(f"Features prepared: {features}")
    
    # Make prediction using the loaded model
    prediction = model.predict(features)
    print(f"Prediction made: {prediction}")
    
    # Output (0 or 1)
    output = int(prediction[0])
    print(f"Output: {output}")
    
    return render_template('index.html', prediction_text=f'Diabetes Prediction: {output}')

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
