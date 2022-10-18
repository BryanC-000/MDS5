"""
This file is the main file used to run the web application.
"""

####### IMPORTS #######
import numpy as np
import tensorflow as tf
import cv2
import io
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from flask_sqlalchemy import SQLAlchemy

####### Initialise Global Variables & Configure the Application #######
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://mds5:postgres@localhost:5432/mds5"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class image(db.Model):
    """
    Schema for the database table for images   
    """
    id = db.Column(db.Integer, primary_key=True)
    img = db.Column(db.LargeBinary, nullable=False)
    name = db.Column(db.String, nullable=False, primary_key = True)
    mimetype = db.Column(db.String, nullable=False)

# File paths of the trained deep learning models
path1='model_training/saved_model/InceptionResnetV2.h5'
path2="model_training/saved_model/InceptionV3.h5"
path3="model_training/saved_model/ResNet50.h5"

# Loading the deep learning models
model1 = tf.keras.models.load_model(path1)
model2 = tf.keras.models.load_model(path2)
model3 = tf.keras.models.load_model(path3)

####### Application Functions #######
@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders the upload page and ensures that the database is empty.
    """ 
    delete_images()
    return render_template('upload.html')

@app.route('/aboutus', methods = ['GET', 'POST'])
def about_us():
    """
    Renders the about us page.
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

    Input: An image 
    Output: The classification of the input image into one of Benign, InSitu, Invasive or Normal
    """
    vals = ["Benign","InSitu","Invasive","Normal"]

    IMG = []
    RESIZE = 299
    image = np.asarray(image.convert("RGB")) # change to RGB
    img = cv2.resize(image, (RESIZE,RESIZE)) # resize image
    IMG.append(np.array(img))
    data = np.array(IMG)
    pred1 = model1.predict(data) 

    if all:
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
        db.session.add(img)
        db.session.commit()

        return render_template('beforeresults.html') 

def get_image(db=db): 
    """
    Retrieves the last image uploaded from the database
    
    Input: The database to retrieve the image from
    Output: The image retrieved from the database
    """
    retrieved_img = db.session.query(image).order_by(image.id.desc()).first()
    if not retrieved_img:
        return 'No image found', 404
    ret_img = Image.open(io.BytesIO(retrieved_img.img))
    return ret_img 

def delete_images(db=db):
    """
    Deletes all images in the database
    
    Input: The database to delete the images from
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
