"""
Used Car Price Predictor - Flask API
Model: Gradient Boosting Machine (GBM)
Author: Keith TANG
Performance: R² = 0.8064 | MAE = $5,132 | RMSE = $10,378

Run: python app.py
API: POST /predict with JSON body
"""

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder='.')

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'gbm_pipeline.pkl')
pipeline = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Build input dataframe
        input_df = pd.DataFrame([{
            'Brand': data.get('brand', 'Toyota'),
            'Year': int(data.get('year', 2018)),
            'UsedOrNew': data.get('usedOrNew', 'USED'),
            'Transmission': data.get('transmission', 'Automatic'),
            'DriveType': data.get('driveType', 'Front'),
            'FuelType': data.get('fuelType', 'Unleaded'),
            'FuelConsumption': float(data.get('fuelConsumption', 8.0)),
            'Kilometres': float(data.get('kilometres', 50000)),
            'CylindersinEngine': data.get('cylindersInEngine', '4 cyl'),
            'BodyType': data.get('bodyType', 'SUV'),
            'Doors': data.get('doors', '4 Doors'),
            'Seats': data.get('seats', '5 Seats'),
            'State': data.get('state', 'NSW'),
            'Cylinders': float(data.get('cylinders', 4.0)),
            'Capacity': float(data.get('capacity', 2.0)),
        }])
        
        log_pred = pipeline.predict(input_df)[0]
        price = float(np.exp(log_pred))
        
        # Confidence interval (±1 std of log prediction ≈ ±25% range)
        low = float(np.exp(log_pred - 0.25))
        high = float(np.exp(log_pred + 0.25))
        
        return jsonify({
            'predicted_price': round(price, -2),
            'range_low': round(low, -2),
            'range_high': round(high, -2),
            'log_prediction': round(log_pred, 4),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'Gradient Boosting Machine',
        'r2': 0.8064,
        'mae': 5132,
        'rmse': 10378
    })

if __name__ == '__main__':
    print("🚗 Used Car Price Predictor API")
    print("   Model: Gradient Boosting Machine")
    print("   R² = 0.8064 | MAE = $5,132 | RMSE = $10,378")
    print("   Running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
