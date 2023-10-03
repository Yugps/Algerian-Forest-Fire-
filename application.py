from flask import Flask,request,render_template
import pandas as pd

import pickle 
model=pickle.load(open('lassocv.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict_datapoint',methods=['GET','POST']) # simply give both post and get here for error free execution 
def predictor(): # remember the name of the function should be same as the one mentioned in the template file for this function
    if request.method=='POST':
        day=int(request.form.get('day')) # get or simply .form can also be used 
        month=int(request.form.get('month'))
        year=int(request.form.get('year'))
        rh=int(request.form.get('RH'))
        ws=int(request.form.get('Ws'))
        rain=float(request.form.get('Rain'))
        ffmc=float(request.form.get('FFMC'))
        dmc=float(request.form.get('DMC'))
        dc=float(request.form.get('DC'))
        isi=float(request.form.get('ISI'))
        bui=float(request.form.get('BUI'))
        fwi=float(request.form.get('FWI'))
        fire=int(request.form.get('fire'))
        not_fire=int(request.form.get('not fire'))
        input_params=pd.DataFrame([day,month,year,rh,ws,rain,ffmc,dmc,dc,isi,bui,fwi,fire,not_fire]).T
        scaled_data=scaler.transform(input_params)
        pred=model.predict(scaled_data)
        return render_template('home.html',result=pred[0])
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)
    
