from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import pandas as pd
import joblib
import sys
from sklearn.pipeline import Pipeline
from src.Disease_classification.exception import CustomException
from src.Disease_classification.utils import load_obj

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionDataset(BaseModel):
    Disease: str
    Fever: str
    Cough: str
    Fatigue: str
    Difficulty_Breathing: str
    Age: int
    Gender: str
    Blood_Pressure: str
    Cholesterol_Level: str

# Load pre-trained objects
Disease_label = joblib.load('artifacts\Dise.jobllib')
preprocessor_obj = load_obj('artifacts/preprocessor.pkl')
mulla = load_obj('artifacts/model.pkl')

# Define the model pipeline
model_pipe = Pipeline(steps=[
    ("Preprocessor", preprocessor_obj),
    ("model", mulla)
])

@app.get("/", response_class=HTMLResponse)
def home():
    return "Welcome to my web page:::###"

@app.get("/pred", response_class=HTMLResponse)
async def get_prediction_page(request: Request):
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

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
    uvicorn.run(app,port=8000)
