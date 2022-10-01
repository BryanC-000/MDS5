import cv2
import sys
import os
import tempfile
# adding Folder_2 to the system path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import models
from werkzeug.utils import secure_filename
from db import db, db_init
import unittest
from PIL import Image
from main import app
from flask import Flask, render_template, request

class InitialiseDatabaseTest(unittest.TestCase):
    def test_new_image(self):
        """
        GIVEN an image
        WHEN an image is uploaded
        THEN check the image id, name, type and mimetype
        """
        pic = Image.open("TESTING/test_load_images/PNG/1_normal.png")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()
        img = models.Img(img = pic, mimetype = mimetype, name = filename)
        
        # Check names/values
        assert img.mimetype == 'image/png'
        assert img.name == 'TESTING_test_load_images_PNG_1_normal.png'
        pic.close()

    def test_database(self):
        # TODO: create an app & test if img.db is there
        tester = os.path.exists("img.db")
        self.assertTrue(tester)


class UploadTest(unittest.TestCase):
    # set up a new temp database for per test
    
    def setUp(self):
        from flask_sqlalchemy import SQLAlchemy

        app = Flask(__name__)
        # self.app.config['TESTING'] = True
        # self.app.config['UPLOAD_FOLDER'] = 'uploaded'
        # self.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///img.db"
        # self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        self.db = db
        self.db.init_app(app)
        app.app_context().push()
        # Creates the tables if the db doesnt already exist
        with app.app_context():
            self.db.create_all()

    # destroy the temp database for per test
    def tearDown(self):
        app = Flask(__name__)
        self.db.init_app(app)
        with app.app_context():
            self.db.drop_all()
            
    def test_add_image(self):
        self.db.session.query(models.Img).delete() # delete all from testing db first
        self.db.session.commit()
        from PIL import Image
        pic = Image.open("TESTING/test_load_images/PNG/1_normal.png")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()

        import io
        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format=pic.format)
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        image = models.Img(img = imgByteArr, mimetype = mimetype, name = filename)

        self.db.session.add(image)
        self.db.session.commit()
        self.assertTrue(models.Img.query.count() == 1)  # case 1

    def test_get_del_img(self):
        pass

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule( sys.modules[__name__] )
    unittest.TextTestRunner(verbosity=3).run( suite )
    # unittest.main()     



    
