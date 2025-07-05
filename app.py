from flask import Flask,request,jsonify,render_template
import pickle 
import pandas as pd
app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template("index.html")

def get_cleaned_data(form_data):
    gestation=float(form_data['gestation'])
    parity=int(form_data['parity'])
    age=float(form_data['age'])
    weight=float(form_data['weight'])
    height=float(form_data['height'])
    smoke=float(form_data['smoke'])

    cleaned_data={"gestation":[gestation],
                   "parity":[parity],
                    "age":[age],
                    "height":[height],
                    "weight":[weight],
                     "smoke":[smoke] 
            }
    return cleaned_data
  

##define your endpoint
@app.route("/predict",methods=['POST'])
def get_prediction():
    #get data from user
    baby_data_form=request.form #getting information from form
    baby_data=get_cleaned_data(baby_data_form)
    #convert into data frame
    baby_df=pd.DataFrame(baby_data)
    
    #load machine learning trained model
    with open("model.pkl","rb") as obj:
        model=pickle.load(obj)

    #make prediction on user data
    prediction= model.predict(baby_df)

    prediction=round(float(prediction),2)

    #return response in json format
    response={"Prediction":prediction}
    return render_template('index.html',prediction=prediction)



if __name__=='__main__':
    app.run(debug=True)
