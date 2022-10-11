import time
start = time.time()

import json
import itertools
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from model_building import Dataset_loader, create_labels, train_valid_test_split

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

def kfold(k, x_train, y_train, model):
    """
    K fold cross validation
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
    Plot a confusion matrix
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

def plot_training_progress(history):
    """
    Plot graph of training progress
    """
    # with open("model_training/history.json", 'r') as f:
    #     history = json.loads(f.read())
    history_df = pd.DataFrame(history.history)
    history_df[['accuracy', 'val_accuracy','loss', 'val_loss']].plot()  

def main(PATH):
    """
    Does all the evaluation of the model given a filepath
    """
    # Load dataset
    IMG_SIZE = 299
    benign_data = np.array(Dataset_loader('model_training/Photos/Benign',IMG_SIZE))
    insitu_data = np.array(Dataset_loader('model_training/Photos/InSitu',IMG_SIZE))
    invasive_data = np.array(Dataset_loader('model_training/Photos/Invasive',IMG_SIZE))
    normal_data = np.array(Dataset_loader('model_training/Photos/Normal',IMG_SIZE))
    
    X, Y = create_labels([benign_data, insitu_data,invasive_data, normal_data], 4)
    x_train, x_val, x_test, y_train, y_val, y_test = train_valid_test_split(X, Y, 0.7)

    # Load model
    model = tf.keras.models.load_model(PATH)

    # Plot training progress
    # history = json.loads("model_training/history.json")
    # plot_training_progress(history)

    # Evaluate performance metrics
    accuracy, precision, recall, f1 = evaluate_model(model, PATH, x_test, y_test)
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
    PATH = "saved_model/somemodel.h5"
    main(PATH)   

    