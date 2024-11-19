from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import numpy as np

# Load model
try:
    model = joblib.load('diabetes_model.pkl')
except FileNotFoundError:
    raise RuntimeError("Model file 'diabetes_model.pkl' not found. Ensure it's in the correct location.")

# Create a FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing, restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the input data model using Pydantic
class PredictionRequest(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int

    @validator('glucose', 'insulin', 'bmi', 'blood_pressure', 'skin_thickness', 'diabetes_pedigree_function', 'age')
    def check_positive(cls, value):
        if value < 0:
            raise ValueError('Value must be non-negative')
        return value

@app.get("/")
def home():
    return {"message": "Welcome to the Health Outcome Prediction API!"}

@app.post("/predict/")
def predict(request: PredictionRequest):
    # Prepare the input data
    data = np.array([[request.pregnancies, request.glucose, request.blood_pressure, 
                      request.skin_thickness, request.insulin, request.bmi, 
                      request.diabetes_pedigree_function, request.age]])

    try:
        # Make a prediction
        prediction = model.predict(data)
        # Map prediction result
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")