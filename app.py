from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract input data from the form
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        self_employed = request.form['self_employed']
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])
        property_area = request.form['property_area']

        # Map categorical inputs
        gender = 1 if gender == "Male" else 0
        married = 1 if married == "Yes" else 0
        education = 1 if education == "Graduate" else 0
        self_employed = 1 if self_employed == "Yes" else 0
        property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

        # Create feature array
        features = np.array([[gender, married, dependents, education, self_employed,
                              applicant_income, coapplicant_income, loan_amount,
                              loan_amount_term, credit_history, property_area]])
        
        # Predict using the model
        prediction = model.predict(features)

        # Return result
        result = "Eligible" if prediction == 1 else "Not Eligible"
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
