from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained Random Forest Regression model
with open('random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the HTML form
    BHK = int(request.form['BHK'])
    Size = float(request.form['Size'])
    Area_Type = int(request.form['Area_Type'])
    City = int(request.form['City'])
    Furnishing_Status = int(request.form['Furnishing_Status'])
    Tenant_Preferred = int(request.form['Tenant_Preferred'])
    Bathroom = int(request.form['Bathroom'])
    Point_of_Contact = int(request.form['Point_of_Contact'])

    # Create a numpy array with the input values
    input_data = np.array([[BHK, Size, Area_Type, City, Furnishing_Status, Tenant_Preferred, Bathroom, Point_of_Contact]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Use the trained Random Forest Regression model to predict the rent
    predicted_rent = random_forest_model.predict(input_data_scaled)

    return render_template('result.html', predicted_rent=predicted_rent[0])

if __name__ == '__main__':
    app.run(debug=True)
