from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained ML model (relative to app.py)
model = joblib.load('random_forest_model.pkl')



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        MonthlyCharges = float(request.form['MonthlyCharges'])
        Tenure = float(request.form['Tenure'])
        TotalCharges = float(request.form['TotalCharges'])

        features = np.array([[MonthlyCharges, Tenure, TotalCharges]])
        prediction = model.predict(features)

        output = "Customer will CHURN" if prediction[0] == 1 else "Customer will NOT churn"
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
