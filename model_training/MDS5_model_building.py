from distutils.command.build import build
from pathlib import Path
import time
start = time.time()

import shutil
import os
import cv2

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from keras import layers
from keras.applications import ResNet50, InceptionV3, InceptionResNetV2
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

def Dataset_loader(DIR, RESIZE, sigmaX=10):
	IMG = []
	read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
	for IMAGE_NAME in tqdm(os.listdir(DIR)):
		PATH = os.path.join(DIR,IMAGE_NAME)
		_, ftype = os.path.splitext(PATH)
		if ftype == ".tif":                     # Transfer tif images to array images 
			img = read(PATH)                       # read image path & convert to RGB
			img = cv2.resize(img, (RESIZE,RESIZE)) # resize image
			IMG.append(np.array(img))              # convert image to numpy array & append to array of images
	return IMG

def create_labels(array, num_class):
	"""
	Creates labels and shuffles the data given an array of images belonging to different classes
	"""
	labels = [0] * len(array)
	for i in range(len(array)):
		labels[i] = np.full(len(array[i]), i)
	
	# Merge data
	X = np.concatenate(tuple([data for data in array]), axis = 0)
	Y = np.concatenate(tuple([label for label in labels]), axis = 0)

	# Shuffle data
	s = np.arange(X.shape[0])
	np.random.seed(1234)
	np.random.shuffle(s)
	X = X[s]
	Y = Y[s]
	Y = to_categorical(Y, num_classes = num_class)
	return X, Y

def train_valid_test_split(X, Y, train_size, random = 1234):
	"""
	Function to split a given dataset into training, validation and testing datasets where validation and testing portions are the same size
	"""
	# Training set
	x_train, X_remainder, y_train, Y_remainder = train_test_split(
		X, Y, 
		train_size=train_size, 
		random_state=random
		)
	# Validation and Test set
	x_val, x_test, y_val, y_test = train_test_split(
		X_remainder, Y_remainder, 
		train_size=0.5, 
		random_state=random
		)
	return x_train, x_val, x_test, y_train, y_val, y_test

def export_images(x, y, labels, path):
	"""
	Function to save images to a given filepath
	"""
	try:
		shutil.rmtree(path)
	except FileNotFoundError:
		pass
	finally:
		os.mkdir(path)
		for i in range(len(x)):
			data = Image.fromarray(x[i])
			label = labels[np.argmax(y[i])]
			filepath = f"{path}/{i+1}_{label}.png"
			data.save(filepath)

def build_model(backbone, lr=1e-4, num_class = 4):
	"""
	Function to build a model given a backbone model as the feature extractor.
	"""
	model = Sequential()
	model.add(backbone)
	model.add(layers.GlobalAveragePooling2D(name = "final"))
	model.add(layers.Dropout(0.5))
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(num_class, activation='softmax'))

	model.compile(
		loss='categorical_crossentropy',
		optimizer=Adam(lr=lr),
		metrics=['accuracy']
	)

	return model

def train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, filepath):
	"""
	Trains a model given the training and validation datasets along with the required hyperparameters
	"""
	# Data augmentation
	train_generator = ImageDataGenerator(
			zoom_range = 2,  # set range for random zoom
			rotation_range = 90, # set range for image rotation
			horizontal_flip=True,  # randomly flip images
			vertical_flip=True,  # randomly flip images 
		)

	# Model callbacks
	# Learning Rate Reducer
	learn_control = ReduceLROnPlateau(monitor='val_accuracy', patience=5,
									verbose=1,factor=0.2, min_lr=1e-7)

	# Checkpoint
	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

	# Early stopping
	early_checkpoint = EarlyStopping(patience = 10, monitor = 'val_accuracy', mode = "max")   
	history = model.fit(
		train_generator.flow(x_train, y_train, batch_size=batch_size), 
		steps_per_epoch=x_train.shape[0] / batch_size,      
		epochs=epochs,
		validation_data=(x_val, y_val),
		callbacks=[learn_control, checkpoint, early_checkpoint],
		verbose=0
		) 
	
	return history

def evaluate_model(model, weight_path, x_test, y_test):
	"""
	Evaluates the performance of a model given a testing dataset
	"""
	model.load_weights(weight_path) # load the best checkpoint weights
	y_pred = model.predict(x_test)   
	accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)) 
	precision = precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average="macro") 
	recall = recall_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average="macro") 
	f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average="macro") 

	return accuracy, precision, recall, f1

def main(IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, PATH):
	"""
	Executes the entire process of:
	- data loading
	- data preprocessing
	- data splitting
	- model building and training
	- model evaluation
	"""
	# Loading the dataset for each class
	benign_data = np.array(Dataset_loader('Photos/Benign',IMG_SIZE))
	insitu_data = np.array(Dataset_loader('Photos/InSitu',IMG_SIZE))
	invasive_data = np.array(Dataset_loader('Photos/Invasive',IMG_SIZE))
	normal_data = np.array(Dataset_loader('Photos/Normal',IMG_SIZE))

	X, Y = create_labels([benign_data, insitu_data,invasive_data, normal_data], 4)

	x_train, x_val, x_test, y_train, y_val, y_test = train_valid_test_split(X, Y, 0.7)
	# print(x_test.shape)
	export_images(x_test, y_test, labels = ["benign", "insitu", "invasive", "normal"], path = "exported_images")

	myModel = InceptionResNetV2(
		weights='imagenet',
		include_top=False,
		input_shape=(IMG_SIZE,IMG_SIZE,3)
	)

	model = build_model(myModel, lr = LEARNING_RATE, num_class=4)
	history = train_model(model, x_train, y_train, x_val, y_val, BATCH_SIZE, EPOCHS, filepath = PATH)

	accuracy, precision, recall, f1 = evaluate_model(model, PATH, x_test, y_test)
	print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1-score: {f1}")
	tf.keras.models.save_model(model, "saved_model/InceptionResNetV2")    
	model.save("saved_model/InceptionResNetV2.h5")
	# tf.keras.model.save_model("saved_model/InceptionResNetV2.h5")

if __name__ == "__main__":
	IMG_SIZE = 299
	BATCH_SIZE = 16
	EPOCHS = 50
	LEARNING_RATE = 0.0001
	PATH = "weights/weights.InceptionResNetV2.hdf5"
	main(IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, PATH)