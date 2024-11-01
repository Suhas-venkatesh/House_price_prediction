from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('house_price_prediction.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = 1 if request.form['mainroad'] == 'yes' else 0
    guestroom = 1 if request.form['guestroom'] == 'yes' else 0
    basement = 1 if request.form['basement'] == 'yes' else 0
    hotwaterheating = 1 if request.form['hotwaterheating'] == 'yes' else 0
    airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
    parking = int(request.form['parking'])
    prefarea = 1 if request.form['prefarea'] == 'yes' else 0
    furnishingstatus = request.form['furnishingstatus']
    
    # Convert furnishing status to numerical value
    furnishing_map = {'unfurnished': 2, 'semi-furnished': 1, 'furnished': 0}
    furnishingstatus = furnishing_map[furnishingstatus]
    
    # Prepare input data for prediction
    features = np.array([[area, bedrooms, bathrooms, stories, mainroad, guestroom,
                          basement, hotwaterheating, airconditioning, parking,
                          prefarea, furnishingstatus]])
    
    features = scaler.transform(features)
    # Make prediction
    prediction = model.predict(features)
    
    # Return prediction result
    return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
