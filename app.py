from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
import joblib
import sys
from sklearn.pipeline import Pipeline
from src.Disease_classification.exception import CustomException
from base import PredictionDataset

# Assuming load_obj is defined elsewhere in src.Disease_classification.utils
from src.Disease_classification.utils import load_obj

app = FastAPI()

# Load pre-trained objects
Disease_label = joblib.load('artifacts\Dise.jobllib')
preprocessor_obj = load_obj('artifacts\preprocessor.pkl')
mulla = load_obj('artifacts\model.pkl')

# Define the model pipeline
model_pipe = Pipeline(steps=[
    ("Preprocessor", preprocessor_obj),
    ("model", mulla)
])

# Pydantic model for prediction input


@app.get("/")
def home():
    return "Welcome to my web page:::###"

@app.post("/pred")
def do_pred(test_data: PredictionDataset):
    try:
        # Convert input data to DataFrame
        X_test = pd.DataFrame([test_data.dict()])
        X_test['Disease'] = Disease_label.transform(X_test['Disease'])
        
        # Predict using the model pipeline
        prediction = model_pipe.predict(X_test).reshape(-1, 1)
        
        # Determine prediction result
        r = "Positive" if prediction[0] > 0.5 else "Negative"
        return {"Prediction": f"The chances of patient is {r} ###"}
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    test_data = PredictionDataset(
        Disease="Eczema",
        Fever="No",
        Cough="Yes",
        Fatigue="Yes",
        Difficulty_Breathing="No",
        Age=25,
        Gender="Female",
        Blood_Pressure="Normal",
        Cholesterol_Level="Normal"
    )
    
    try:
        # Test the function
        result = do_pred(test_data)
        print(result)
    except CustomException as e:
        print(e)
    
    uvicorn.run(app, port=8000)
