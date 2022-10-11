import unittest
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import model_training.model_building_evaluation as model_building_evaluation

import numpy as np
import os
from tqdm import tqdm
from keras.applications import ResNet50, InceptionV3, InceptionResNetV2
import tensorflow as tf
from unittest.mock import Mock, MagicMock, patch

class BuildModelTest(unittest.TestCase):
    BATCH_SIZE = 1
    EPOCHS = 5
    benign = model_building_evaluation.Dataset_loader('testing/testing_files/test_create_labels/benign', 299)
    invasive = model_building_evaluation.Dataset_loader('testing/testing_files/test_create_labels/invasive', 299)
    insitu = model_building_evaluation.Dataset_loader('testing/testing_files/test_create_labels/insitu', 299)
    normal = model_building_evaluation.Dataset_loader('testing/testing_files/test_create_labels/normal', 299)
    array = [benign,invasive,insitu,normal]
    X, Y = model_building_evaluation.create_labels(array,4)
    x_train, x_val, x_test, y_train, y_val, y_test=model_building_evaluation.train_valid_test_split(X,Y,0.6)
    myModel = InceptionResNetV2(
		weights='imagenet',
		include_top=False,
		input_shape=(299,299,3)
	)
    model = model_building_evaluation.build_model(myModel,0.0001)
    history = model_building_evaluation.train_model(model,x_train,y_train,x_val, y_val, BATCH_SIZE, EPOCHS, filepath = 'testing/testing_files/test_train_model')
    training_progress= model_building_evaluation.plot_training_progress(history)
    
    kfold_res = model_building_evaluation.kfold(3, x_train, y_train, model)
    y_pred = model.predict(x_test)  
    cm = model_building_evaluation.plot_confusion_matrix(y_test, y_pred, ['benign', 'insitu', 'invasive', 'normal'] , title ='Confusion Metrix for Breast Cancer')
    accuracy, precision, recall, f1 = model_building_evaluation.evaluate_model(model, x_test, y_test)

    def test_load_images_multiple_png(self):
        """

        """
        resize = 299
        normal_png_image = model_building_evaluation.Dataset_loader('testing/testing_files/test_load_images/PNG', resize)
        self.assertEqual(len(normal_png_image), 0)

    def test_load_images_multiple_tif(self):
        resize = 299
        normal_tif_image = model_building_evaluation.Dataset_loader('testing/testing_files/test_load_images/TIF', resize)
        self.assertEqual(normal_tif_image[0].shape, (resize,resize,3))
        self.assertEqual(len(normal_tif_image), 5)

    def test_create_labels(self):
        X = np.concatenate(tuple([data for data in self.array]), axis = 0)
        ori_labels = [0] * len(self.array)
        for i in range(len(self.array)):
            ori_labels[i] = np.full(len(self.array[i]), i)
        Y = np.concatenate(tuple([label for label in ori_labels]), axis = 0)
        # after concatenate shud be benign, insitu,invasive,normal...
        self.assertTrue((X!=self.X).any()) # test if the images have been shuffled
        test = []
        for x in range(len(Y)):
            test.append(Y[x]!=np.argmax(self.Y[x]))# test if the labels have been shuffled
        self.assertTrue(any(test))
        for i in self.Y:
            self.assertEqual(len(i),4) # 4 classes so each should only have 4
            for x in i:
                self.assertTrue(x==0.0 or x==1.0) #test if its one-hot encoded

    def test_train_valid_test_split(self):
        # Loading the dataset for each class
        self.assertEqual(self.x_train.shape[0],self.X.shape[0]*0.6)
        self.assertEqual(self.y_train.shape[0],self.Y.shape[0]*0.6)
        self.assertEqual(self.x_test.shape[0],self.X.shape[0]*0.2)
        self.assertEqual(self.y_test.shape[0],self.Y.shape[0]*0.2)
        self.assertEqual(self.x_val.shape[0],self.X.shape[0]*0.2)
        self.assertEqual(self.y_val.shape[0],self.Y.shape[0]*0.2)
        self.assertEqual(self.x_train.shape[0],3)
        self.assertEqual(self.y_train.shape[0],3)
        self.assertEqual(self.x_test.shape[0],1)
        self.assertEqual(self.y_test.shape[0],1)
        self.assertEqual(self.x_val.shape[0],1)
        self.assertEqual(self.x_val.shape[0],1)

    def test_export_images(self):
        # check if naming with label
        # check if the images are saved correctly in the correct path given
        # test try except
        class_labels = ["benign", "insitu", "invasive", "normal"]
        model_building_evaluation.export_images(self.x_test, self.y_test, labels = class_labels, path = 'testing/testing_files/test_exported_images')
        lst = os.listdir("testing/testing_files/test_exported_images") # your directory path
        number_files = len(lst)
        self.assertEqual(number_files,1) # test if the test image exported are correct number
        for IMAGE_NAME in tqdm(os.listdir("testing/testing_files/test_exported_images")):
            only_img_name ,_= IMAGE_NAME.split('.')
            self.assertRegex(only_img_name,r'^([1-3])(_benign|_insitu|_invasive|_normal)$') #test if exported file format is correct

    def test_build_model(self):
        self.assertIsNotNone(self.model)
        layer = self.model.get_layer('dense')
        self.assertEqual(layer.output_shape[1],4) #test if the model have 4 output nodes

    def test_train_model(self):
        self.assertIsNotNone(self.history)

    def test_evaluate_model(self):
        # test range of accuracy
        self.assertTrue(self.accuracy >= 0.0 and self.accuracy <= 1.0)
        # test range of precision
        self.assertTrue(self.precision>= 0.0 and self.precision <= 1.0)
        # test range of recall
        self.assertTrue(self.recall>= 0.0 and self.recall <= 1.0)
        # test range of f1-score
        self.assertTrue(self.f1 >= 0.0 and self.f1 <= 1.0)

    def test_kfold(self):
        self.assertTrue(self.kfold_res >= 0.0 and self.kfold_res <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity = 3)   
     
