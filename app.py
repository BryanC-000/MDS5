import numpy as np
import tensorflow as tf
import cv2
import io
import psycopg2

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

from db import db_init, db
from models import Img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///img.db"
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://lmsbetafbhatex:1f0846d30373368a26ac5e42cc5c7ef84e46ff17e92fae8a497f3e028b2e9cfa@ec2-18-209-78-11.compute-1.amazonaws.com:5432/d5sp11rdepbqda"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

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
    path1='model_training/saved_model/InceptionResnetV2.h5'
    model1 = tf.keras.models.load_model(path1)
    vals = ["Benign","InSitu","Invasive","Normal"]

    IMG = []
    RESIZE = 299
    img = cv2.resize(image, (RESIZE,RESIZE)) # resize image
    IMG.append(np.array(img))
    data = np.array(IMG)
    pred1 = model1.predict(data) 

    if all:
        path2="model_training/saved_model/InceptionV3.h5"
        model2 = tf.keras.models.load_model(path2)
        path3="model_training/saved_model/ResNet50.h5"
        model3 = tf.keras.models.load_model(path3)
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

        img = Img(img = pic.read(), mimetype = mimetype, name = filename)
        db.session.add(img)
        db.session.commit()

        return render_template('beforeresults.html') 

def get_image():
    """
    Retrieves the image inputted from the database
    """
    img = Img.query.filter_by(id=1).first()
    if not img:
        return 'No image found', 404
    image = np.array(Image.open(io.BytesIO(img.img))) 
    return image 

def delete_images():
    """
    Deletes all images in the database
    """
    db.session.query(Img).delete() 
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
    app.run()
