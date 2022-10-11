import numpy as np
import tensorflow as tf
import cv2
import io
import os
import psycopg2

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from db import db_init, db
from models import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///img.db"
# app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://mds5:postgres@localhost:5432/mds5"
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://qwkupolrhbotbb:79788c40d960331a9d2f0bec8c5604b21590da9f18edddb13d9b3bc9891ad104@ec2-52-54-212-232.compute-1.amazonaws.com:5432/d9cp1tc750akj4"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db_init(app)
db = SQLAlchemy(app)
migrate = Migrate(app,db)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "credentials.json"

# path1='model_training/saved_model/InceptionResnetV2.h5'
# path2="model_training/saved_model/InceptionV3.h5"
# path3="model_training/saved_model/ResNet50.h5"

bucket_path = "gs://mds5_bucket_1"
path1 = bucket_path + "/InceptionResNetV2"
path2 = bucket_path + "/InceptionV3"
path3 = bucket_path + "/ResNet50"

model1 = tf.keras.models.load_model(path1)
model2 = tf.keras.models.load_model(path2)
model3 = tf.keras.models.load_model(path3)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders the upload page and ensures that the database is empty
    """ 
    delete_images()
    return render_template('upload.html')

@app.route('/aboutus', methods = ['GET', 'POST'])
def about_us():
    """
    Renders the about us page
    """
    if request.method == 'POST':
        return render_template('aboutus.html')      

@app.route('/aboutthemodel', methods = ['GET', 'POST'])
def about_the_model():
    """
    Renders the about the model page
    """
    if request.method == 'POST':
        return render_template('aboutthemodel.html')        

def model_predict(image, all = False):
    """
    Gets the predicted class for the given image using only proposed model or all models
    """
    
    vals = ["Benign","InSitu","Invasive","Normal"]

    IMG = []
    RESIZE = 299
    img = cv2.resize(image, (RESIZE,RESIZE)) # resize image
    IMG.append(np.array(img))
    data = np.array(IMG)
    pred1 = model1.predict(data) 

    if all:
        read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

        pred2 = model2.predict(data) 

        RESIZE = 224
        img = cv2.resize(image, (RESIZE,RESIZE)) # resize image
        IMG[0] = np.array(img)
        data = np.array(IMG)
        pred3 = model3.predict(data) 

        return str(vals[np.argmax(pred1)]), str(vals[np.argmax(pred2)]), str(vals[np.argmax(pred3)])

    return str(vals[np.argmax(pred1)]), "NONE", "NONE"

@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
    """
    Renders the page after an image is uploaded and adds the image into the database
    """
    if request.method == 'POST':
        pic = request.files['pic'] 

        filename = secure_filename(pic.filename) 
        mimetype = pic.mimetype

        img = image(img = pic.read(), mimetype = mimetype, name = filename)
        # img = Img(mimetype = mimetype, name = filename)
        db.session.add(img)
        db.session.commit()

        return render_template('beforeresults.html') 

def get_image(db=db): # added db argument for testing
    """
    Retrieves the image inputted from the database
    """
    # retrieved_img = db.session.query(image).first()
    retrieved_img = db.session.query(image).order_by(image.id.desc()).first()
    if not retrieved_img:
        return 'No image found', 404
    ret_img = np.array(Image.open(io.BytesIO(retrieved_img.img))) 
    return ret_img 

def delete_images(db=db): # added db argument for testing
    """
    Deletes all images in the database
    """
    db.session.query(image).delete() 
    db.session.commit()

@app.route('/proposedmodelresult', methods = ['GET', 'POST'])
def predict():
    """
    Gets the prediction using the proposed model and displays the results screen
    """
    if request.method == 'POST':
        img = get_image()
        val1, val2, val3 = model_predict(img, all = False) # only proposed model predicted result
        return render_template('results.html', pred1=val1, pred2=val2, pred3=val3)   

@app.route('/allmodelresult', methods = ['GET', 'POST'])
def predict_all():
    """
    Gets the prediction using all models and displays the results screen
    """
    if request.method == 'POST':
        img = get_image()
        val1, val2, val3 = model_predict(img, all = True) # all models' predicted results
        return render_template('results.html', pred1=val1, pred2=val2, pred3=val3)                       

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run()
