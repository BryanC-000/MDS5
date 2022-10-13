import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../model_training'))
from model_building_evaluation import Dataset_loader, create_labels, train_valid_test_split, evaluate_model
import numpy as np
import tensorflow as tf
from tensorflow import keras

def test_models(IMG_SIZE, MODEL_PATH):
    benign_data = np.array(Dataset_loader('model_training/Photos/Benign',IMG_SIZE))
    insitu_data = np.array(Dataset_loader('model_training/Photos/InSitu',IMG_SIZE))
    invasive_data = np.array(Dataset_loader('model_training/Photos/Invasive',IMG_SIZE))
    normal_data = np.array(Dataset_loader('model_training/Photos/Normal',IMG_SIZE))

    X, Y = create_labels([benign_data, insitu_data, invasive_data, normal_data], 4)

    x_train, x_val, x_test, y_train, y_val, y_test = train_valid_test_split(X, Y, 0.7)

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model: ", MODEL_PATH)
    for i in range(5):
        accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)
        print("Performance metrics of model:")
        print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1-score: {f1}")
    


    
if __name__ == "__main__":
    IMG_SIZES = [299, 299, 224]
    PATHS = ["model_training/saved_model/InceptionResnetV2.h5", "model_training/saved_model/InceptionV3.h5", "model_training/saved_model/ResNet50.h5"]
    
    for i in range(3):
        test_models(IMG_SIZES[i], PATHS[i])


