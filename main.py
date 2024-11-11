from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# CORS settings to allow your frontend (adjust accordingly)
origins = [
    "http://127.0.0.1:8000",  # The origin where your Flutter app is running (adjust if necessary)
    "http://localhost",  # Local development for Flutter (adjust if necessary)
    "http://localhost:3000",  # Flutter web running locally (adjust if needed)
    "https://fast-api-vv4w.onrender.com",  # The URL where your backend is hosted
]

# Adding CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

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
