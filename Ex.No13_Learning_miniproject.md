# Ex.No: 13 Machine Learning Miniproject  
### DATE: 04/11/24                                                                         
### REGISTER NUMBER : 212222040176
### AIM: 
To build a predictive model that accurately classifies weather types based on meteorological data (precipitation, temperature, and wind).###  Algorithm:
### Program:
```
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('new_data.csv')

# Separate features and target variable
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = df['weather']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and scaler for later use
with open('weather_prediction_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")

# Import necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('seattle-weather.csv')

# Define a mapping for weather types to numeric values
weather_mapping = {
    'drizzle': 0,
    'rain': 1,
    'sun': 2,
    'snow': 3,
    'fog': 4
}

# Map the 'weather' column to numeric values using the defined mapping
df['weather'] = df['weather'].map(weather_mapping)

# Save the new dataset
df.to_csv('new_data.csv', index=False)

print("Weather column converted to numbers (including fog) and saved as 'new_data.csv'")

# Import necessary libraries
import pickle
import numpy as np

# Load the saved model and scaler
with open('weather_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Sample input data (example values for 'precipitation', 'temp_max', 'temp_min', 'wind')
sample_input = np.array([[10.0, 15.0, 5.0, 3.5]])

# Scale the sample input using the loaded scaler
sample_input_scaled = scaler.transform(sample_input)

# Predict the weather
predicted_weather_numeric = model.predict(sample_input_scaled)

# Map numeric prediction back to the weather type
weather_mapping = {
    0: 'drizzle',
    1: 'rain',
    2: 'sun',
    3: 'snow',
    4: 'fog'
}
predicted_weather = weather_mapping[predicted_weather_numeric[0]]

print("Predicted Weather:", predicted_weather)
```
### Output:
![Screenshot 2024-11-12 172019](https://github.com/user-attachments/assets/32f4c00a-05ff-40eb-bebb-ac38a8dbf1f3)


### Result:
Thus the system was trained successfully and the prediction was carried out.
