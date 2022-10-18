"""
This file is a testing file that is used to test the functionality of the database
"""

####### IMPORTS #######
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from app import image, db
from werkzeug.utils import secure_filename
import unittest
import PIL
from PIL import Image
import app
import io

####### Testing Class and Test Functions #######
class InitialiseDatabaseTest(unittest.TestCase):
    def setUp(self):
        """
        Function to define instructions that will be executed before each test method
        
        Input: none

        Output: none
        """
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.app.config['UPLOAD_FOLDER'] = 'uploaded'
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
        """
        Function to define instructions that will be executed after each test method
        
        Input: none

        Output: none
        """
        self.app = Flask(__name__)
        self.db.init_app(self.app)
        with self.app.app_context():
            self.db.drop_all()

    def test_new_image(self):
        """
        Function that tests if the new uploaded image is correctly saved with the correct id, name and mimetype.
        
        Input: none

        Output: none
        """
        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()
        IMG = image(id=1, img = pic, mimetype = mimetype, name = filename)
        
        # Check names & values
        self.assertEquals(IMG.id, 1)
        self.assertEquals(IMG.mimetype, 'image/tiff')
        self.assertEquals(IMG.name, "model_training_Photos_Normal_n001.tif")
        pic.close()
    
    def test_database(self):
        """
        Function that tests if the database engine exists.
        
        Input: none

        Output: none
        """
        tester = os.path.exists(("/"+str(self.db.engine.url).strip("postgresql://mds5:postgres@localhost:5432/mds5")))
        self.assertTrue(tester)
    
    def test_add_image(self):
        """
        Function that tests if the image file that is added to the database table is successfully stored in the database.
        
        Input: none

        Output: none
        """
        self.db.session.query(image).delete()
        self.db.session.commit()

        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()

        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format=pic.format)
        imgByteArr = imgByteArr.getvalue()
        IMG = image(id=1, img = imgByteArr, mimetype = mimetype, name = filename)

        # Add image
        self.db.session.add(IMG)
        self.db.session.commit()
        self.assertEquals(self.db.session.query(image).count(), 1)
        pic.close()

    def test_get_image_valid(self):
        """
        Function that tests the get_image() function from app.py, to see if the image is successfully retreived from the database
        
        Input: none

        Output: none
        """
        self.db.session.query(image).delete() 
        self.db.session.commit()
        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()
        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format=pic.format)
        imgByteArr = imgByteArr.getvalue()
        IMG = image(id=1, img = imgByteArr, mimetype = mimetype, name = filename)
        self.db.session.add(IMG)
        
        # Get image  
        img1 = app.get_image(self.db) 
        self.assertEquals(type(img1), PIL.TiffImagePlugin.TiffImageFile)
        pic.close()

    def test_get_image_invalid(self):
        """
        Function that tests the get_image() function from app.py, to see if there is no image retrieved after attempting a retrieval from an empty database
        
        Input: none

        Output: none
        """
        self.db.session.query(image).delete()
        
        img1 = app.get_image(self.db) 
        self.assertEquals(img1, ('No image found', 404))

    def test_delete_images(self):
        """
        Function that tests the delete_images(), to see if all existing images in the database table have successfully been removed from the database
        
        Input: none

        Output: none
        """
        self.db.session.query(image).delete()
        self.db.session.commit()
        pic = Image.open("model_training/Photos/Normal/n001.tif")
        filename = secure_filename(pic.filename) 
        mimetype = "image/"+pic.format.lower()
        imgByteArr = io.BytesIO()
        pic.save(imgByteArr, format=pic.format)
        imgByteArr = imgByteArr.getvalue()
        IMG = image(id=1, img = imgByteArr, mimetype = mimetype, name = filename)
        self.db.session.add(IMG)
        self.assertEquals(self.db.session.query(image).count(), 1)

        # Delete all images
        app.delete_images(self.db)
        self.assertEquals(self.db.session.query(image).count(), 0)
        

if __name__ == "__main__":
    unittest.main(verbosity = 3)   



    
