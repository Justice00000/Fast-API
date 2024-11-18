from fastapi import FastAPI
import joblib
import numpy as np

# Load your model
model = joblib.load('diabetes_model.pkl')

# Create a FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Health Outcome Prediction API!"}

@app.post("/predict/")
def predict(pregnancies: int, glucose: float, blood_pressure: float, skin_thickness: float, insulin: float, bmi: float, dpf: float, age: int):
    # Prepare the input data
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    # Make a prediction
    prediction = model.predict(data)
    # Map prediction result
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return {"prediction": result}