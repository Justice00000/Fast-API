from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load model
model_path = "diabetes_model.pkl"
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError(f"Model file '{model_path}' not found. Ensure it's in the correct location.")

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class PredictionRequest(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int
    outcome: int  # New field: user-provided actual outcome (1 = Diabetic, 0 = Non-Diabetic)

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
    # Prepare input data
    data = np.array([[request.pregnancies, request.glucose, request.blood_pressure, 
                      request.skin_thickness, request.insulin, request.bmi, 
                      request.diabetes_pedigree_function, request.age]])

    try:
        # Make a prediction
        prediction = model.predict(data)
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        
        # Retrain the model using this new data point
        retrain_model(data, request.outcome)
        
        return {"prediction": result, "message": "Prediction made and model retrained."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def retrain_model(new_data, new_label):
    try:
        # Load existing training data
        try:
            data_df = pd.read_csv("training_data.csv")
        except FileNotFoundError:
            data_df = pd.DataFrame(columns=["Pregnancies", "Glucose", "BloodPressure", 
                                            "SkinThickness", "Insulin", "BMI", 
                                            "DiabetesPedigreeFunction", "Age", "Outcome"])

        # Convert new data to DataFrame and add it to training set
        new_df = pd.DataFrame(new_data, columns=["Pregnancies", "Glucose", "BloodPressure", 
                                                 "SkinThickness", "Insulin", "BMI", 
                                                 "DiabetesPedigreeFunction", "Age"])
        new_df["Outcome"] = new_label

        # Append and save updated dataset
        data_df = pd.concat([data_df, new_df], ignore_index=True)
        data_df.to_csv("training_data.csv", index=False)

        # Extract features and labels
        X = data_df.drop(columns=["Outcome"])
        y = data_df["Outcome"]

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train a new model
        new_model = LogisticRegression()
        new_model.fit(X_scaled, y)

        # Save the updated model
        joblib.dump(new_model, model_path)

    except Exception as e:
        print(f"Retraining failed: {str(e)}")
