import joblib
import pandas as pd
from flask import Flask, request, Response
from healthguard import HealthGuard 

# Load the trained model
MODEL_FILE = '..repos/deploy/model/final_model.joblib'
try:
    loaded_model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    raise Exception(f"Model file not found at {MODEL_FILE}")

# Initialize the Flask app
app = Flask(__name__)

# Define the API route
@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    # Parse the JSON input data
    try:
        test_json = request.get_json(force=True)
    except:
        return Response('Invalid JSON input', status=400, mimetype='application/json')

    # Check if the input data is valid
    if not test_json:
        return Response('No input data provided', status=400, mimetype='application/json')

    if isinstance(test_json, dict):
        test_raw = pd.DataFrame([test_json])
    elif isinstance(test_json, list) and all(isinstance(x, dict) for x in test_json):
        test_raw = pd.DataFrame(test_json)
    else:
        return Response('Invalid input data format', status=400, mimetype='application/json')

    # Initialize the pipeline
    pipeline = HealthGuard()

    try:
        # Data cleaning
        df1 = pipeline.data_cleaning(test_raw)

        # Feature engineering
        df2 = pipeline.feature_engineering(df1)

        # Data preparation
        df3 = pipeline.data_preparation(df2)

        # Make predictions
        df_response = pipeline.get_prediction(loaded_model, test_raw, df3)

    except Exception as e:
        return Response(str(e), status=500, mimetype='application/json')

    # Convert the results to JSON and return them
    response_json = df_response.to_json(orient='records', date_format='%Y-%m-%d %H:%M:%S')
    return Response(response_json, status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('127.0.0.1')
