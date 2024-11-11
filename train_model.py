import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
data = pd.read_csv('Housing.csv')
print(data.head())
print(data.columns)

# Select features (X) and target (y) based on your dataset
# Select features (replace these columns with the ones you find relevant)
X = data[['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]  # Example feature columns
y = data['price']  # Target column

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model to a file
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)