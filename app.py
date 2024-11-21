from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('diabetes.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

SCALE_FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction',
        'Age','NewBMI_Obesity 1','NewBMI_Obesity 2','NewBMI_Obesity 3','NewBMI_Overweight','NewBMI_Underweight',
        'NewInsulinScore_Normal','NewGlucose_Low','NewGlucose_Normal','NewGlucose_Overweight','NewGlucose_Secret']

@app.route('/')
def home() :
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def pred() :
    try :
        if not request.json :
            return jsonify({'Error' : "Response Data is not successfully sent"})
        
        d9 = False;d10 = False;d11 = False;d12 = False;d13 = False
        d14 = False;d15 = False;d16 = False;d17 = False;d18 = False
        data = request.json
        print(data)
        d1 = float(data['pregnancy'])
        d2 = float(data['glucose'])
        d3 = float(data['BP'])
        d4 = float(data['SkinThickness'])
        d5 = float(data['Insulin'])
        d6 = float(data['BMI'])
        d7 = float(data['Pedigree'])
        d8 = float(data['Age'])
        
        if d6 < 18.5:
            d13 = True
        elif 18.5 <= d6 <= 24.9:
            d12 = True
        elif 25.0 <= d6 <= 29.9:
            d9 = True
        elif 30.0 <= d6 <= 34.9:
            d10 = True
        elif d6 > 34.9:
            d11 = True
           
        if d2 <= 70:
            d15 = True
        elif 70 < d2 <= 99:
            d16 = True
        elif 100 <= d2 <= 126:
            d17 = True
        elif d2 > 126:
            d18 = True

        d14 = True if (16 <= d5 <= 116) else False
        
        input_list =scaler.transform(pd.DataFrame([[d1,d2,d3,d4,d5,d6,d7,d8]],columns=SCALE_FEATURES))
        cat_bools = list([d9,d10,d11,d12,d13,d14,d15,d16,d17,d18])
        input_list = list(input_list[0])
        input_list = input_list + cat_bools
        print(f"\n\n {input_list} \n")
        
        pred = model.predict([input_list])[0]
        pred = int(pred)
        print(f"Prediction : {pred} {type(pred)}")
        return jsonify({'prediction' : pred,'values' : [input_list]})
        
    except Exception as e :
        print(e)
        return jsonify({'prediction' : e,})


if __name__ == "__main__" :
    
    print("Type is : >>>>>",type(model))
    app.run(debug=True, port=8000)