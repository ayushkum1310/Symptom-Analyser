from pydantic import BaseModel


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
    