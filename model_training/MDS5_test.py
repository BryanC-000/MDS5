import unittest
import MDS5_model_building
import numpy as np
import os
from tqdm import tqdm
from keras.applications import ResNet50, InceptionV3, InceptionResNetV2
import tensorflow as tf

class BuildModelTest(unittest.TestCase):
    benign = []


    def test_load_images(self):
        resize = 100
        normal_png_image = MDS5_model_building.Dataset_loader('model_training/TESTING/test_load_images/PNG', resize)
        self.assertEqual(len(normal_png_image), 0)
        normal_tif_image = MDS5_model_building.Dataset_loader('model_training/TESTING/test_load_images/TIF', resize)
        self.assertEqual(normal_tif_image[0].shape, (resize,resize,3))
        self.assertEqual(len(normal_tif_image), 1)

    def test_create_labels(self):
        benign = MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/benign', 299)
        invasive = MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/invasive', 299)
        insitu = MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/insitu', 299)
        normal = MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/normal', 299)
        array = [benign,invasive,insitu,normal]
        X = np.concatenate(tuple([data for data in array]), axis = 0)
        
        ori_labels = [0] * len(array)
        for i in range(len(array)):
            ori_labels[i] = np.full(len(array[i]), i)
        Y = np.concatenate(tuple([label for label in ori_labels]), axis = 0)
        # after concatenate shud be benign, insitu,invasive,normal...
        data, label = MDS5_model_building.create_labels(array)
        self.assertTrue((X!=data).any()) # test if the images have been shuffled
        test = []
        for x in range(len(Y)):
            test.append(Y[x]!=np.argmax(label[x]))# test if the labels have been shuffled
        self.assertTrue(any(test))
        for i in label:
            self.assertEqual(len(i),4) # 4 classes so each should only have 4
            for x in i:
                self.assertTrue(x==0.0 or x==1.0) #test if its one-hot encoded
                self.assertEqual((type(np.argmax(x))),np.int64) # test if only 1 integer is being returned
        return data, label

    def test_train_valid_test_split(self):
        	# Loading the dataset for each class
        IMG_SIZE=299
        # X,Y = self.test_create_labels()
        benign_data = np.array(MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/benign',IMG_SIZE))
        insitu_data = np.array(MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/invasive',IMG_SIZE))
        invasive_data = np.array(MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/insitu',IMG_SIZE))
        normal_data = np.array(MDS5_model_building.Dataset_loader('model_training/TESTING/test_create_labels/normal',IMG_SIZE))
        X, Y = MDS5_model_building.create_labels([benign_data, insitu_data,invasive_data, normal_data])
        x_train, x_val, x_test, y_train, y_val, y_test=MDS5_model_building.train_valid_test_split(X,Y,0.7)
        self.assertEqual(x_train.shape[0],X.shape[0]*0.7)
        self.assertEqual(y_train.shape[0],Y.shape[0]*0.7)
        self.assertEqual(x_test.shape[0],X.shape[0]*0.15)
        self.assertEqual(y_test.shape[0],Y.shape[0]*0.15)
        self.assertEqual(x_val.shape[0],X.shape[0]*0.15)
        self.assertEqual(y_val.shape[0],Y.shape[0]*0.15)

    def test_export_images(self):
        # check if naming with label
        # check if the images are saved correctly in the correct path given
        # test try except
        _, _, x_test, _, _, y_test = self.test_train_valid_test_split()
        class_labels = ["benign", "insitu", "invasive", "normal"]
        MDS5_model_building.export_images(x_test, y_test, labels = class_labels, path = 'model_training/TESTING/test_exported_images')
        lst = os.listdir("model_training/TESTING/test_exported_images") # your directory path
        number_files = len(lst)
        self.assertEqual(number_files,3) # test if the test image exported are correct number
        for IMAGE_NAME in tqdm(os.listdir("model_training/TESTING/test_exported_images")):
            only_img_name ,_= IMAGE_NAME.split('.')
            regex = f"^([1-9]|[1-5][0-9]|60)_{class_labels[0]}|{class_labels[1]}|{class_labels[2]}|{class_labels[3]}$"
            self.assertRegex(only_img_name,r'^([1-9]|[1-5][0-9]|60)(_benign|_insitu|_invasive|_normal)$') #test if exported file format is correct


    def build_model(self):
        myModel = InceptionResNetV2(
		weights='imagenet',
		include_top=False,
		input_shape=(299,299,3)
	)
        model = MDS5_model_building.build_model(myModel,0.0001)
        return model

    def test_build_model(self):
        model = self.build_model()
        model.summary() #compare model summary with inceptionResNetv2 architecture paper?
        layer = model.get_layer('dense')
        self.assertEqual(layer.output_shape[1],4) #test if the model have 4 output nodes
        return model

    def test_train_model(self):
        x_train, x_val, x_test, y_train, y_val, y_test = self.test_train_valid_test_split()
        model = self.build_model()
        history = MDS5_model_building.train_model(model, x_train, y_train, x_val, y_val, BATCH_SIZE, EPOCHS, filepath = 'model_training/TESTING/test_train_model')
        model = tf.keras.models.load_model('model_training/TESTING/test_train_model')
        # load back model check if usable from same pth
        m
        self.assertIsNotNone(history)
        pass

    def test_evaluate_model(self):
        pass

if __name__ == "__main__":
    BATCH_SIZE = 1
    EPOCHS = 40
    unittest.main()      
