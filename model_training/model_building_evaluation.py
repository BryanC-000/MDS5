"""
This file shows how the deep learning model is built and evaluated across different matrices
"""

####### IMPORTS #######
from distutils.command.build import build
import shutil
import os
import cv2
import json
import itertools

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from tensorflow import keras
from keras import layers
from keras.applications import ResNet50, InceptionV3, InceptionResNetV2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

####### Model Building Functions #######
def Dataset_loader(DIR, RESIZE):
	"""
	Function to load images for a given directory
	
	Input: DIR, which is the existing breast histology dataset file path 
	       Resize, which The new image size

	Output: IMG, which is an array of images loaded as numpy arrays
	"""
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
	Function to create labels and shuffle the data given an array of images belonging to different classes

	Input: Array: An array of images belonging to different classes
		   num_class: The number of classes to classify the breast image into
	Output: X: Shuffled data
	        Y: Shuffled label
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
	Input: X: The shuffled data
		   Y: The shuffled labels of the data encoded in one-hot
           Train size: The size to split the training data
           Random: seed value (Default = 1234)

	Output: x_train: The training data
	        y_train: The label of the training set
			x_val: The validation data
			y_val: The label of the validation set
			x_test: The testing data
			y_test: the label of the testing set
			
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
	Function to save the exported images into a specific filepath
	Input: x: The data (images as numpy array)
		   y: The labels encoded in one-hot 
		   labels: The sets of label value in a list
		   path: The file path to save the given images
	Output: -
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

def build_model(backbone, lr=1e-4, opt = "adam", include = False, num_class = 4):
	"""
	Function to build a model given a backbone model as the feature extractor.
	Input: Backbone: The backbone architecture 
		   Learning rate: The learning rate to be used in our optimizer
		   Optimizer: The optimizer to be used
		   Include: Boolean (Default = False where True = Include the dropout layer and the batch normalisation layer and False = Don't include) 
		   Num_classes: The number of classes to classify the prediction into
	Output: The model built with the specified parameters
	"""
	model = Sequential()
	model.add(backbone)
	model.add(layers.GlobalAveragePooling2D(name = "final"))
	if include:
		model.add(layers.Dropout(0.5))
		model.add(layers.BatchNormalization())
	model.add(layers.Dense(num_class, activation='softmax', name = "output"))
	
	if opt == 'adam':
		optimizer = keras.optimizers.Adam(lr)
	elif opt == 'nadam':
		optimizer = keras.optimizers.Nadam(lr)
	elif opt == 'adagrad':
		optimizer = keras.optimizers.Adagrad(lr)
	elif opt == 'rmsprop':
		optimizer = keras.optimizers.RMSprop(lr)
	elif opt == 'adadelta':
		optimizer = keras.optimizers.Adadelta(lr)
	else:
		optimizer = keras.optimizers.SGD(lr)   

	model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
	
	return model

def train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, filepath, verbose = 1):
	"""
	Function to train a model given the training and validation datasets along with the required hyperparameters
	Input: Model: The deep learning model built 
		   X_train: The training data 
	       Y_train: The label of the training data
           X_val: The validation data 
           Y_val: The label of the validation data 
           Batch_size: The number of training examples utilised in one iteration
           Epochs:  The number of times the learning algorithm will work through the entire training dataset
           Filepath: The file path to checkpoint and save the model to
	Output: The training history of the model
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
									verbose=verbose,factor=0.2, min_lr=1e-7)

	# Checkpoint
	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=verbose, save_best_only=True, mode='max')

	# Early stopping
	early_checkpoint = EarlyStopping(patience = 7, monitor = 'val_accuracy', mode = "max")   
	history = model.fit(
		train_generator.flow(x_train, y_train, batch_size=batch_size), 
		steps_per_epoch=x_train.shape[0] / batch_size,      
		epochs=epochs,
		validation_data=(x_val, y_val),
		callbacks=[learn_control, checkpoint, early_checkpoint],
		verbose=0
		) 
	
	return history

####### Model Evaluation Functions #######
def evaluate_model(model, x_test, y_test):
	"""
	Function to evaluate the performance of a model given a testing dataset
	Input: Model: The deep learning model built
           X_test: The testing dataset
           Y_test: The label of the testing dataset 
	Output: Accuracy:  A ratio of correctly predicted observation to the total observations
	        Precision: The ratio of correctly predicted positive observations to the total predicted positive observations.
			Recall: The ratio of correctly predicted positive observations to the all observations in actual class 
			F1-score: The weighted average of Precision and Recall
	"""
	y_pred = model.predict(x_test)   
	accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)) 
	precision = precision_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average="macro") 
	recall = recall_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average="macro") 
	f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), average="macro") 

	return accuracy, precision, recall, f1

def kfold(k, x_train, y_train, model):
    """
    Function to perform K fold cross validation
	Input: K: The number of folds 
           X_train: The training data 
           Y_train: The label of the training data 
           Model: The deep learning model built 
	Output: The average accuracy score of all folds
    """
    kf = KFold(n_splits=k, random_state=None)
 
    acc_score = []
    prec_score = []
    recall_score = []
    f1_score = []

    for train_index , test_index in kf.split(x_train):
        x_train_kf , x_test_kf = x_train[train_index],x_train[test_index]
        y_train_kf , y_test_kf = y_train[train_index] , y_train[test_index]
        y_test_kf = np.argmax(y_test_kf, axis=1) 
        
        model.fit(x_train_kf,y_train_kf)
        pred_values = model.predict(x_test_kf)
        pred_values=np.argmax(pred_values, axis=1)
        acc = accuracy_score(pred_values , y_test_kf)
        prec = precision_score(pred_values , y_test_kf, average='micro')
        report = classification_report(y_test_kf, pred_values) 
        print(report)
        
        acc_score.append(acc)
        prec_score.append(prec)

    avg_acc_score = sum(acc_score)/k

    return avg_acc_score

def plot_confusion_matrix(y_test, y_pred, classes, title, normalize = False):
    """
    Function to plot a confusion matrix
	Input: y_test: The actual label of the testing set
	       y_pred: The predicted label of the testing set
		   classes: The number of classes
		   Title: The title of the plot
		   Normalize: Boolean whether to normalize the confusion matrix

	Output: A confusion matrix plotted
    """
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return cm

def plot_training_accuracies(history):
    """
    Function to plot graph of training progress - accuracies
	Input: The history of the model training
	Output: -
    """
    history_df = pd.DataFrame(history.history)
    history_df[['accuracy', 'val_accuracy']].plot()

def plot_training_losses(history):
    """
    Function to plot graph of training progress - losses
	Input: The history of the model training
	Output: -
    """
    history_df = pd.DataFrame(history.history)
    history_df[['loss', 'val_loss']].plot()

####### Main Function #######
def build_evaluate_model(DATA, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, OPTIMIZER, INCLUDE, WEIGHT_PATH, SAVE_PATH, BASE_MODEL):
	"""
	Function to build and evaluates a model
	Input:  DATA: x_train: The training data
	       	      y_train: The label of the training set
				  x_val: The validation data
				  y_val: The label of the validation set
				  x_test: The testing data
				  y_test: the label of the testing set
			IMG_SIZE: The size of the image to be fit into the model (resize)
			BATCH_SIZE: The number of training examples utilised in one iteration
           	EPOCHS:  The number of times the learning algorithm will work through the entire training dataset
			LEARNING_RATE,: The learning rate to be used in our optimizer
		    OPTIMIZER: The optimizer to be used
		    INCLUDE: Boolean (Default = False where True = Include the dropout layer and the batch normalisation layer and False = Don't include) 
		    WEIGHT_PATH: The file path to checkpoint and save the weights to
			SAVE_PATH: The path to save the model into
			BASE_MODEL: The name of the backbone architecture

	Output: -
	"""
	x_train, x_val, x_test, y_train, y_val, y_test = DATA
	if BASE_MODEL == "InceptionResnetV2":
		myModel = InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(IMG_SIZE,IMG_SIZE,3))
	elif BASE_MODEL == "InceptionV3":
		myModel = InceptionV3(weights='imagenet',include_top=False,input_shape=(IMG_SIZE,IMG_SIZE,3))
	else:	
		myModel = ResNet50(weights='imagenet',include_top=False,input_shape=(IMG_SIZE,IMG_SIZE,3))

	model = build_model(myModel, lr = LEARNING_RATE, opt = OPTIMIZER, include = INCLUDE, num_class=4)
	print("Training model")
	history = train_model(model, x_train, y_train, x_val, y_val, BATCH_SIZE, EPOCHS, filepath = WEIGHT_PATH)

	print("Saving model")
	with open('model_training/history.json', 'w') as f:
		json.dump(str(history.history), f)
	model.save(SAVE_PATH)

	# Evaluate performance metrics
	accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)
	print("Performance metrics of model")
	print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1-score: {f1}")

    # K Fold
	print("K-Fold cross validation")
	avg_accuracy = kfold(5, x_train, y_train, model)
	print('Avg accuracy : {}'.format(avg_accuracy))

    # Confusion Matrix
	y_pred = model.predict(x_test)
	cm = plot_confusion_matrix(y_test, y_pred, ['benign', 'insitu', 'invasive', 'normal'] , title ='Confusion Metrix for Breast Cancer')
	print("Confusion Matrix")
	print(cm)
	
if __name__ == "__main__":
	# Defining parameters
	IMG_SIZE = 299
	BATCH_SIZE = 16
	EPOCHS = 50
	LEARNING_RATE = 0.0001
	OPTIMIZER = "adam"
	INCLUDE_BN_DROPOUT = True
	WEIGHT_PATH = "model_training/weights/weights.InceptionResnetV2.hdf5"
	SAVE_PATH = "model_training/saved_model/InceptionResnetV2.h5"
	BASE_MODEL = "InceptionResnetV2"

	# Loading the dataset for each class
	benign_data = np.array(Dataset_loader('model_training/Photos/Benign', IMG_SIZE))
	insitu_data = np.array(Dataset_loader('model_training/Photos/InSitu',IMG_SIZE))
	invasive_data = np.array(Dataset_loader('model_training/Photos/Invasive',IMG_SIZE))
	normal_data = np.array(Dataset_loader('model_training/Photos/Normal',IMG_SIZE))

	X, Y = create_labels([benign_data, insitu_data,invasive_data, normal_data], 4)

	x_train, x_val, x_test, y_train, y_val, y_test = train_valid_test_split(X, Y, 0.7)
	DATA = (x_train, x_val, x_test, y_train, y_val, y_test)
	export_images(x_test, y_test, labels = ["benign", "insitu", "invasive", "normal"], path = "exported_images")

	build_evaluate_model(DATA, IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, OPTIMIZER, INCLUDE_BN_DROPOUT, WEIGHT_PATH, SAVE_PATH, BASE_MODEL)