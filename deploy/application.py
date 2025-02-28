from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

application = app  # For deployment platforms

# Load the model and scaler
with open('deploy/models/model.pkl', 'rb') as file:
    log_model = pickle.load(file)
with open('deploy/models/scale.pkl', 'rb') as file:
    scale = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form.get('age'))
            bp = float(request.form.get('bp'))
            sg = float(request.form.get('sg'))
            al = float(request.form.get('al'))
            su = int(request.form.get('su'))
            rbc = int(request.form.get('rbc'))
            pc = int(request.form.get('pc'))
            bgr = float(request.form.get('bgr'))
            bu = float(request.form.get('bu'))
            sc = float(request.form.get('sc'))
            sod = float(request.form.get('sod'))
            hemo = float(request.form.get('hemo'))
            pcv = float(request.form.get('pcv'))
            wbcc = float(request.form.get('wbcc'))
            rbcc = float(request.form.get('rbcc'))
            htn = int(request.form.get('htn'))
            dm = int(request.form.get('dm'))
            ane = int(request.form.get('ane'))

            # Create data array
            data = [age, bp, sg, al, su, rbc, pc, bgr, bu, sc, sod, hemo, pcv, wbcc, rbcc, htn, dm, ane]
            
            # Scale and predict
            new_scaled_data = scale.transform([data])
            result = log_model.predict(new_scaled_data)

            # Return prediction as plain text
            prediction = "Positive" if result[0] == 1 else "Negative"
            return f"Your result: {prediction}"
        except ValueError as e:
            return f"Error: Invalid input - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
        return render_template('predict.html')
    else:
        return render_template('predict.html')
if __name__ == '__main__':
    app.run(debug=True)