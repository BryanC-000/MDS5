import unittest
import pytest
import sys
import os
import io
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from app import app, db, delete_images

@pytest.fixture(scope='module')
def test_client():
    flask_app = app

    # Create a test client using the Flask application configured for testing
    with flask_app.test_client() as testing_client:
        # Establish an application context
        with flask_app.app_context():
            db.create_all()
            yield testing_client  # this is where the testing happens!
            db.drop_all()

def test_home_page(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/' page is requested (GET)
    THEN check that the response is valid
    """
    response = test_client.get('/')
    assert response.status_code == 200

def test_home_page_post(test_client):
    """
    GIVEN a Flask application
    WHEN the '/' page is posted to (POST)
    THEN check that the response is valid
    """
    response = test_client.post('/')
    assert response.status_code == 200

def test_about_us_page(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/aboutus' page is requested (GET)
    THEN check that the response returns '500'
    """
    response = test_client.get('/aboutus')
    assert response.status_code == 500

def test_about_us_page_post(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/aboutus' page is posted to (POST)
    THEN check that the response is valid
    """
    response = test_client.post('/aboutus')
    assert response.status_code == 200

def test_about_the_model_page(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/aboutthemodel' page is requested (GET)
    THEN check that the response returns '500'
    """
    response = test_client.get('/aboutthemodel')
    assert response.status_code == 500

def test_about_the_model_page_post(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/aboutthemodel' page is posted to (POST)
    THEN check that the response is valid
    """
    response = test_client.post('/aboutthemodel')
    assert response.status_code == 200

def test_before_results_page(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/uploaded' page is requested (GET)
    THEN check that the response returns '500'
    """
    response = test_client.get('/uploaded')
    assert response.status_code == 500

def test_before_results_page_post(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/uploaded' page is posted to (POST)
    THEN check that the response is valid
    """
    # Test if an '200' response is returned when a valid image/data is posted
    img = "testing/testing_files/test_load_images/PNG/4_normal.png"
    img_data = open(img, "rb")
    data = {"pic": (img_data, "image.png")}
    
    response = test_client.post(
        '/uploaded',
        data=data,
        buffered=True,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200

    # Test if an '400' bad request response is returned when an invalid image/data is posted
    img_data = open(img, "rb")
    
    response = test_client.post(
        '/uploaded',
        data=img_data,
        buffered=True,
        content_type="multipart/form-data",
    )
    assert response.status_code == 400

def test_proposed_model_results_page(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/proposedmodelresult' page is requested (GET)
    THEN check that the response returns '500'
    """
    response = test_client.get('/proposedmodelresult')

    assert response.status_code == 500

def test_proposed_model_results_page_post(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/proposedmodelresult' page is requested (POST)
    THEN check that the response is valid
    """
    # test after adding image
    # image added
    img = "testing/testing_files/test_load_images/PNG/4_normal.png"
    img_data = open(img, "rb")
    data = {"pic": (img_data, "image.png")}
    
    response = test_client.post(
        '/uploaded',
        data=data,
        buffered=True,
        content_type="multipart/form-data",
    )

    response = test_client.post('/proposedmodelresult')
    assert response.status_code == 200

    # test after deleting images
    delete_images(db)
    response = test_client.post('/proposedmodelresult')
    assert response.status_code == 500
    
def test_all_model_results_page(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/allmodelresult' page is requested (GET)
    THEN check that the response returns '500'
    """
    response = test_client.get('/allmodelresult')

    assert response.status_code == 500

def test_all_model_results_page_post(test_client):
    """
    GIVEN a Flask application configured for testing
    WHEN the '/allmodelresult' page is requested (POST)
    THEN check that the response is valid
    """
    # test after adding image
    # image added
    img = "testing/testing_files/test_load_images/PNG/4_normal.png"
    img_data = open(img, "rb")
    data = {"pic": (img_data, "image.png")}
    
    response = test_client.post(
        '/uploaded',
        data=data,
        buffered=True,
        content_type="multipart/form-data",
    )

    response = test_client.post('/allmodelresult')
    assert response.status_code == 200

    # test after deleting images
    delete_images(db)
    response = test_client.post('/allmodelresult')
    assert response.status_code == 500




