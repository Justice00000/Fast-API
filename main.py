from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load your model
model = joblib.load('diabetes_model.pkl')

# Create a FastAPI app
app = FastAPI()

# Define the input data model using Pydantic
class PredictionRequest(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    dpf: float
    age: int

@app.get("/")
def home():
    return {"message": "Welcome to the Health Outcome Prediction API!"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    # Prepare the input data
    data = np.array([[request.pregnancies, request.glucose, request.blood_pressure, 
                      request.skin_thickness, request.insulin, request.bmi, 
                      request.dpf, request.age]])
    # Make a prediction
    prediction = model.predict(data)
    # Map prediction result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return {"prediction": result}