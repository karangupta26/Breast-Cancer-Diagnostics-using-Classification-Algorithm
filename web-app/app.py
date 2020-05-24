
from flask import Flask, flash, request, redirect, url_for,render_template,send_from_directory
from werkzeug.utils import secure_filename

from PIL import Image
from ResNet50 import resnet50_extractor
from vgg16 import vgg16_extractor
from xception import xception_extractor
from joblib import load
import os
import io

import sqlite3

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
SAMPLE_FOLDER='SAMPLE_FOLDER/'
tools='tools/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = SAMPLE_FOLDER

def feature_extacter(img):
    VGG16_feature=vgg16_extractor(img)
    Xception_feature=xception_extractor(img)
    ResNet50_feature=resnet50_extractor(img)
    return VGG16_feature,Xception_feature,ResNet50_feature

def predict_magnification(feature):
    model=load('models/LR_Models_ResNet50_Magnification.joblib')
    mag=model.predict(feature)
    mag_proba=model.predict_proba(feature)
    return mag,mag_proba

def predict_cancerclass(mag,VGG16_feature,Xception_feature,ResNet50_feature):
    if mag==40:
        model=load('models/SVM_Models_ResNet50_Magnification_40.joblib')
        cc=model.predict(ResNet50_feature)
    elif mag==100:
        model=load('models/LR_Models_VGG16_Magnification_100.joblib')
        cc=model.predict(VGG16_feature)
    elif mag==200:
        model=load('models/SVM_Models_ResNet50_Magnification_200.joblib')
        cc=model.predict(ResNet50_feature)
    elif mag==400:
        model=load('models/LR_Models_Xception_Magnification_400.joblib')
        cc=model.predict(Xception_feature)
    return cc

def predict_cancertype(cancerclass,Xception_feature,ResNet50_feature):
    if cancerclass==1:
        model=load('models/LR_Models_Xception_CancerType_Benign.joblib')
        ct=model.predict(Xception_feature)
    else:
        model=load('models/LR_Models_ResNet50_CancerType_Malignant.joblib')
        ct=model.predict(ResNet50_feature)
    return ct
def save_to_db(patientname,patientcontactno,doctorname,hospitalname,histopathlogicalsample,magnification,cancerclass,cancertype):
    conn=sqlite3.connect("webappdb.db")
    c=conn.cursor()
    c.execute("INSERT INTO Medicalrecord"\
                "(PatientName,PatientContactNumber,DoctorName,HospitalName,HistopathologicalSample,Magnification,CancerClass,CancerType) VALUES"\
                "(?,?,?,?,?,?,?,?)",(patientname,patientcontactno,doctorname,hospitalname,histopathlogicalsample,magnification,cancerclass,cancertype))
    conn.commit()
    conn.close()

################################################
# Error Handling
################################################

@app.errorhandler(400)
def FUN_400(error):
    return render_template("error.html"), 40

@app.errorhandler(405)
def FUN_405(error):
    return render_template("error.html"), 405

@app.errorhandler(413)
def FUN_413(error):
    return render_template("error.html"), 413

@app.errorhandler(500)
def FUN_500(error):
    return render_template("error.html"), 500



@app.route("/")
def hello():
    print("Server Started.")
    return render_template("Home.html")


@app.route("/analyse",methods=['POST','GET'])
def form():
    #form=MedicalRecord(request.form)
    if request.method=='POST':
        patientname=request.form['patientname']
        patientcontactno=request.form['patientcontactno']
        doctorname=request.form['doctorname']
        hospitalname=request.form['hospitalname']
        img=request.files['sample']
        print(type(patientcontactno))
        if len(patientcontactno)<10 or len(patientcontactno)>10:
            return render_template("error.html")
        if patientname=='' or patientcontactno==None or doctorname=='' or hospitalname=='':
            return render_template("error.html")

        #Checking for unique value
        conn=sqlite3.connect("webappdb.db")
        c=conn.cursor()
        c.execute("""SELECT PatientContactNumber FROM MedicalRecord
                   WHERE PatientContactNumber=?""",
                (patientcontactno,))
        result=c.fetchone()
        if result:
            return render_template("error.html")
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename)))
        VGG16_feature,Xception_feature,ResNet50_feature=feature_extacter(img)
        print(len(VGG16_feature))
        print(len(Xception_feature))
        print(len(ResNet50_feature))
        VGG16_feature=VGG16_feature.reshape(1,-1)
        Xception_feature=Xception_feature.reshape(1,-1)
        ResNet50_feature=ResNet50_feature.reshape(1,-1)

        mag,mag_proba=predict_magnification(ResNet50_feature)
        print(mag)
        print(type(mag))
        print(mag_proba)

        cancerclass=predict_cancerclass(mag,VGG16_feature,Xception_feature,ResNet50_feature)
        print(cancerclass)

        cancertype=predict_cancertype(cancerclass,Xception_feature,ResNet50_feature)
        print(cancertype)
        print(type(cancertype))
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        print(full_filename)
        print(type(full_filename))
        pilimage=Image.open(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename)))
        print(type(pilimage))
        dbimg=io.BytesIO()
        pilimage.save(dbimg,format='PNG')
        dbimg=dbimg.getvalue()

        if mag==40:
            mag_db='40'
        elif mag==100:
            mag_db='100'
        elif mag==200:
            mag_db='200'
        else:
            mag_db='400'

        if cancerclass==1:
            cancerclass_db='Bengin'
        else:
            cancerclass_db='Malignant'

        if cancertype==11:
            cancertype_db='Adenosis'
        elif cancertype==12:
            cancertype_db='Fibro Adenoma'
        elif cancertype==13:
            cancertype_db='Tubulor Adenoma'
        elif cancertype==14:
            cancertype_db='Phyllodes Tumor'
        elif cancertype==21:
            cancertype_db='Ductol Carcinoma'
        elif cancertype==22:
            cancertype_db='Lobular Carcinoma'
        elif cancertype==23:
            cancertype_db='Mucious Carcinoma'
        else:
            cancertype_db='Pappillary Carcinoma'

        save_to_db(patientname,patientcontactno,doctorname,hospitalname,sqlite3.Binary(dbimg),mag_db,cancerclass_db,cancertype_db)
    return render_template("analyse.html",magnification=mag_db ,cancer_class=cancerclass_db,cancer_type=cancertype_db,image_name=img.filename,patientname=patientname,patientcontactno=patientcontactno,doctorname=doctorname)

@app.route("/analyse/<filename>")
def send_image(filename):
    return send_from_directory("SAMPLE_FOLDER",filename)

if __name__ == '__main__':
    app.run(debug=True)
