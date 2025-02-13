#
# from flask import Flask ,render_template,request
# import pandas as pd
# import pickle
# import numpy as np
#
# with open("RidgeModel.pkl", "rb") as f:
#     model = pickle.load(f)
# print("Model loaded successfully!")
#
#
# app= Flask(__name__)
# data=pd.read_csv('Cleaned_data.csv',  encoding='utf-8' )
#
# pipe=pickle.load(open("RidgeModel.pkl",'rb'))
#
# @app.route('/')
# def index():
#
#     locations=sorted(data['location'].unique())
#     return render_template( 'index.html',locations= locations)
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     location = request.form.get('location')
#     bhk = request.form.get('bhk')
#     bath = request.form.get('bath')
#     sqft = request.form.get('total_sqft')
#     print(f" DEBUG:Location:= {repr(location)}, BHK:= {repr(bhk)}, Bath:={repr(bath)}, Sqft: ={repr(sqft)}")
#
#     #print(f" DEBUG:Location:= {location}, BHK:= {bhk}, Bath:={bath}, Sqft: ={sqft}")
#     #convert numeric values to proper formate
#     # Check if any value is None
#     if None in [location, bhk, bath, sqft]:
#         return "Error: Missing form data. Please ensure all fields are filled."
#         # Convert numeric values properly
#     try:
#         bhk = int(bhk)
#         bath = int(bath)
#         sqft = float(sqft)
#     except ValueError:
#         return "Error: Invalid input format. BHK and Bath should be integers, Sqft should be a number."
#
#     #print(location,bhk,bath,sqft)
#     put=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
#     prediction=pipe.predict(put)[0] *1e5
#
#     return  str(np.round(prediction,2))
#
#
# if __name__ == "__main__":
#
#    app.run(debug=True,port=5001)


from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model with error handling
try:
    with open("RidgeModel.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Handle the case where the model doesn't load

# Load data
data = pd.read_csv('Cleaned_data.csv', encoding='utf-8')

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Error: Model not loaded correctly."

    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    if None in [location, bhk, bath, sqft]:
        return "Error: Missing form data."

    try:
        bhk = int(bhk)
        bath = int(bath)
        sqft = float(sqft)
    except ValueError:
        return "Error: Invalid input format."

    put = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = model.predict(put)[0] * 1e5

    return str(np.round(prediction, 2))

# Fix port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
