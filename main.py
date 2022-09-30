import numpy as np
import tensorflow as tf
import cv2
import io

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

from db import db_init, db
from models import Img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///img.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

@app.route('/', methods=['GET', 'POST'])
def upload_f(): 
    return render_template('upload.html')

def model_predict(image, all = False):
    path1="saved_model/InceptionResNetV2"
    model1 = tf.keras.models.load_model(path1)
    vals = ["Benign","InSitu","Invasive","Normal"]

    IMG = []
    RESIZE = 299
    img = cv2.resize(image, (RESIZE,RESIZE)) # resize image
    IMG.append(np.array(img))
    data = np.array(IMG)
    pred1 = model1.predict(data) 

    if all:
        path2="saved_model/InceptionV3"
        model2 = tf.keras.models.load_model(path2)
        path3="saved_model/ResNet50"
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

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        pic = request.files['pic']

        filename = secure_filename(pic.filename) 
        mimetype = pic.mimetype

        img = Img(img = pic.read(), mimetype = mimetype, name = filename)
        db.session.add(img)
        db.session.commit()

        return render_template('beforeresults.html') 

def get_and_del_img():
    img = Img.query.filter_by(id=1).first()
    if not img:
        return 'No image found', 404

    image = np.array(Image.open(io.BytesIO(img.img))) 
    Img.query.filter_by(id=1).delete()
    db.session.commit()
    return image 

@app.route('/proposedmodelresult', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = get_and_del_img()
        val1, val2, val3 = model_predict(img, all = False) # only proposed model predicted result
        return render_template('results.html', pred1=val1, pred2=val2, pred3=val3)   
        return None

@app.route('/allmodelresult', methods = ['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        img = get_and_del_img()
        val1, val2, val3 = model_predict(img, all = True) # all models' predicted results
        return render_template('results.html', pred1=val1, pred2=val2, pred3=val3)        

@app.route('/aboutus', methods = ['GET', 'POST'])
def aboutus():
    if request.method == 'POST':
        return render_template('aboutus.html')      

@app.route('/aboutthemodel', methods = ['GET', 'POST'])
def aboutthemodel():
    if request.method == 'POST':
        return render_template('aboutthemodel.html')                    

if __name__ == '__main__':
    app.run()
