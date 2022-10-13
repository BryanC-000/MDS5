import cv2
import sys
import os
import tempfile
# adding Folder_2 to the system path
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from models import image
from werkzeug.utils import secure_filename
from db import db, db_init
import unittest
from PIL import Image
from app import app, delete_images
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import app
import io
import numpy as np


# Mocking
class InitialiseDatabaseTest(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['UPLOAD_FOLDER'] = 'uploaded'
        # self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///testing.db"
        self.app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://mds5:postgres@localhost:5432/mds5"
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        self.db = db
        self.db.init_app(self.app)
        self.app.app_context().push()
        # Creates the tables
        with self.app.app_context():
            self.db.create_all()

    # destroy the temp database tables for per test
    def tearDown(self):
        self.app = Flask(__name__)
        self.db.init_app(self.app)
        with self.app.app_context():
            self.db.drop_all()

    def test_new_image(self):
        """
        GIVEN an image
        WHEN an image is uploaded
        THEN check the image id, name and mimetype is correct
        """
        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()
        IMG = image(id=1, img = pic, mimetype = mimetype, name = filename)
        
        # Check names/values
        self.assertEquals(IMG.id, 1)
        self.assertEquals(IMG.mimetype, 'image/tiff')
        self.assertEquals(IMG.name, "model_training_Photos_Normal_n001.tif")
        pic.close()
    
    def test_database(self):
        """
        GIVEN a database file path
        THEN check that the database file path exists
        """
        tester = os.path.exists(("/"+str(self.db.engine.url).strip("postgresql://mds5:postgres@localhost:5432/mds5")))
        self.assertTrue(tester)
    
    def test_add_image(self):
        """
        GIVEN a database and image file
        WHEN the image file is converted to an image object and added to the database table
        THEN check that the image object is indeed stored
        """
        self.db.session.query(image).delete() # delete all from the db first
        self.db.session.commit()

        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()

        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format=pic.format)
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        IMG = image(id=1, img = imgByteArr, mimetype = mimetype, name = filename)

        self.db.session.add(IMG)
        self.db.session.commit()
        self.assertEquals(self.db.session.query(image).count(), 1)
        pic.close()

    def test_get_image_valid(self):
        """
        GIVEN a database with an image object
        WHEN get_image() is used to retrieve the image
        THEN check that the image object image is retrieved
        """
        self.db.session.query(image).delete() # delete all from testing db first
        # put an image in the db first
        self.db.session.commit()
        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()
        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format=pic.format)
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        IMG = image(id=1, img = imgByteArr, mimetype = mimetype, name = filename)
        self.db.session.add(IMG)
        
        img1 = app.get_image(self.db) # get image  
        self.assertEquals(type(img1), np.ndarray)
        pic.close()

    def test_get_image_invalid(self):
        """
        GIVEN a database without image objects
        WHEN get_image() is used to retrieve the image
        THEN check that the image object image is retrieved
        """
        self.db.session.query(image).delete() # delete all from testing db first
        
        img1 = app.get_image(self.db) 
        self.assertEquals(img1, ('No image found', 404))

    def test_delete_images(self):
        """
        GIVEN a database with an image object
        WHEN delete_images() is used to delete all image objects from the database
        THEN check that there are no more images in the database
        """
        self.db.session.query(image).delete() # delete all from testing db first
        # put an image in the db first
        self.db.session.commit()
        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()
        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format=pic.format)
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        IMG = image(id=1, img = imgByteArr, mimetype = mimetype, name = filename)
        self.db.session.add(IMG)

        # delete all images
        app.delete_images(self.db)
        self.assertEquals(self.db.session.query(image).count(), 0)
        

if __name__ == "__main__":
    # suite = unittest.TestLoader().loadTestsFromModule( sys.modules[__name__] )
    # unittest.TextTestRunner(verbosity=1).run( suite )
    unittest.main(verbosity = 3)   



    
