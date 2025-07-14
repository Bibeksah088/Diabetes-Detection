from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize Flask App
app = Flask(__name__)

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_features = scaler.transform([features])
        prediction = model.predict(final_features)[0]
        output = "Diabetic" if prediction == 1 else "Not Diabetic"
        return render_template('index.html', prediction_text=f'Result: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
