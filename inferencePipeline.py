import joblib
import pandas as pd
import numpy as np
def inference_pipeline(dataset):
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("best_model.pkl")
    numeric_features = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']]

    scaled_data = scaler.transform(numeric_features)
    dataset = pd.get_dummies(dataset,columns=['BMI_category'],dtype=int)

    prediction = model.predict(dataset)
    return prediction
