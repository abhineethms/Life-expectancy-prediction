from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from data_preprocessing import preprocess_data

app = Flask(__name__)

model = joblib.load('models/trained_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    #df = df.drop(['Country', 'Year', 'Status'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    
    # Get column names for prediction
    features = df.columns
    
    # Make sure column order is consistent with model
    X_scaled = scaler.transform(df)
    
    # Predict using the model
    prediction = model.predict(X_scaled)

    

    
    return jsonify({'life_expectancy': prediction[0].tolist()})

if __name__ == '__main__':
    app.run(debug=True)
