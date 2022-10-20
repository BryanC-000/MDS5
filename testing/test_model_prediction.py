"""
This file is used to test the performance of saved models
"""

####### IMPORTS #######
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../model_training'))
from model_building_evaluation import Dataset_loader, create_labels, evaluate_model
import numpy as np
import tensorflow as tf
import time

####### Functions #######
def test_models(IMG_SIZE, MODEL_PATH):
    """
    Function to test a saved model on the 400 histology images from BACH2018, for five times.

    Input: IMG_SIZE, the image size that the histology images need to be resized to for the specific model being used
           MODEL_PATH, the filepath of the saved model
    Output: -
    """
    benign_data = np.array(Dataset_loader('model_training/Photos/Benign',IMG_SIZE))
    insitu_data = np.array(Dataset_loader('model_training/Photos/InSitu',IMG_SIZE))
    invasive_data = np.array(Dataset_loader('model_training/Photos/Invasive',IMG_SIZE))
    normal_data = np.array(Dataset_loader('model_training/Photos/Normal',IMG_SIZE))

    X, Y = create_labels([benign_data, insitu_data, invasive_data, normal_data], 4)

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model: ", MODEL_PATH)
    for i in range(5):
        start_time = time.time()
        accuracy, precision, recall, f1 = evaluate_model(model, X, Y)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("accuracy: ", accuracy, "precision: ", precision, "recall: ", recall, "f1: ", f1)


if __name__ == "__main__":
    IMG_SIZES = [299, 299, 224]
    PATHS = ["model_training/saved_model/InceptionResnetV2.h5", "model_training/saved_model/InceptionV3.h5", "model_training/saved_model/ResNet50.h5"]    

    for i in range(3):
        test_models(IMG_SIZES[i], PATHS[i])


