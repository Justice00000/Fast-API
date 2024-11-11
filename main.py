from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define the request model (for the input data)
class PredictionRequest(BaseModel):
    area: float
    bedrooms: float
    bathrooms: float
    stories: float
    parking: float

# Define the prediction endpoint
@app.post("/predict")
def predict(data: PredictionRequest):
    # Extract features from the request data
    features = np.array([[data.area, data.bedrooms, data.bathrooms, data.stories, data.parking]])
    
    # Make a prediction using the model
    prediction = model.predict(features)[0]
    
    # Return the prediction as a JSON response
    return {"prediction": prediction}

# Optional: Add a root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the Linear Regression Prediction API"}