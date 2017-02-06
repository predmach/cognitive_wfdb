#! /usr/bin/env python
# Charles Pace

# Copyright (c) 2016 Predictive Machines, LLC

'''
	test.py - hack away

	main 					- the test drive, will run everything

	load_all_data			- extract - physionet, translate - ml features, load - hdf5
	extract_and_stage_ml	- load from hdf5, convert into features

	exampleRandomForest		- build, train, classify, analyze
	exampleNeuralNet		- build, train, classify, analyze
]
'''

from __future__ import print_function, with_statement
# from StringIO import StringIO
import re
import os

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# from sknn.platform import gpu32
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
from sknn.mlp import Classifier as SknnClassifier
from sknn.mlp import Layer as SknnLayer
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from wfdb import *
from features import *
from hyper import *
from learning import *


def exampleKernelDensity(samples, labels, columns):
    ''' KernelDensity
    '''
    pca = PCA(n_components=15, whiten=False)
    print(samples.shape)
    data = pca.fit_transform(samples)

    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.logspace(-1, 1, 20)}

    clf = KernelDensity()
    x_train, x_val, y_train, y_val = \
        train_test_split(samples, labels, test_size=0.2,
                         random_state=np.random.randint(0, 100))

    print('   running hyper parameter search and training...')
    grid = hyperparameter_search(clf, params, x_train, y_train, x_val, y_val, columns)

    return grid


def exampleRandomForest(samples, labels, columns):
    ''' RandForest
    '''
    params = {
        'n_estimators': [11],
        'n_jobs': [-1]
    }

    clf = RandomForestClassifier()  # Let's run RandForest -> verify that
    x_train, x_val, y_train, y_val = \
        train_test_split(samples, labels, test_size=0.2,
                         random_state=np.random.randint(0, 100))

    print('   running hyper parameter search and training...')
    grid = hyperparameter_search(clf, params, x_train, y_train, x_val, y_val, columns)

    return grid


def exampleNeuralNetTfDnn(samples, labels):
    x_train, x_val, y_train, y_val = \
        train_test_split(samples.values, labels.values, test_size=0.20,
                         random_state=np.random.randint(0, 100))

    # convert into 2 columns, isGood, isBad
    from tflearn.data_utils import to_categorical
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)

    '''
            x_train 	: x training data
            y_train 	: y training labels
            x_val 		: x validation data
            y_val 		: y validation labels
    '''

    import tflearn

    # define neural net
    net = tflearn.input_data(shape=[None, x_train.shape[1]])
    net = tflearn.fully_connected(net, 50, activation='relu')
    net = tflearn.fully_connected(net, 200, activation='relu')
    net = tflearn.fully_connected(net, 10, activation='tanh')
    net = tflearn.fully_connected(net, 2, activation='softmax');
    net = tflearn.regression(net, learning_rate=0.005)

    # define model
    model = tflearn.DNN(net)

    # train
    model.fit(x_train, y_train, show_metric=True, batch_size=x_train.shape[0], n_epoch=30,
              # validation_set=(x_val,y_val), validation_batch_size=x_val.shape[0]
              )

    # print accuracy statistics
    result = model.evaluate(x_train, y_train, batch_size=x_train.shape[0])
    accuracy_score = result[0]
    print("Train Accuracy: %f" % accuracy_score)

    result = model.evaluate(x_val, y_val, batch_size=x_val.shape[0])
    accuracy_score = result[0]
    print("Vdalidation Accuracy: %f" % accuracy_score)

    # print detail statistics
    import sklearn.metrics as skmetric

    print("----- validation")
    pred = model.predict(x_val)
    pred = np.around(np.asarray(pred))
    print("classification")
    print(classification_report(np.asarray(y_val.argmax(axis=1)), np.asarray(pred.argmax(axis=1))))
    print("confusion matrix")
    print(skmetric.confusion_matrix(np.asarray(y_val.argmax(axis=1)), np.asarray(pred.argmax(axis=1))))

    print("----- training")
    pred = model.predict(x_train)
    pred = np.around(np.asarray(pred))
    print("classification")
    print(classification_report(y_train.argmax(axis=1), pred.argmax(axis=1)))
    print("confusion matrix")
    print(skmetric.confusion_matrix(np.asarray(y_train.argmax(axis=1)), np.asarray(pred.argmax(axis=1))))


def exampleNeuralNet(samples, labels, columns):
    # loads it all to start (do once):
    # load_all_data('data/raw_data', 'data/stage_data', 'data/hdf5', 'data/hdf5/mit-bih.hdf')

    ''' NeuralNet
        TODO:
            testTrainSplit - uniform selection
            unequal trining set (all normal data too)
            autoencode on all data
            set to verbose
    '''
    '''
    params={
        'learning_rate': [0.005, 0.001],
        'hidden0__units': [10, 50, 100],
        'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"],
        'hidden1__units': [10, 50, 200, 434],
        'hidden1__type': ["Rectifier", "Sigmoid", "Tanh"],
        'hidden2__units': [10, 50, 200, 434],
        'hidden2__type': ["Rectifier", "Sigmoid", "Tanh"]
    }
    '''

    params = {
        'learning_rate': [0.005],
        'hidden0__type': ['Rectifier'],
        'hidden0__units': [50],
        'hidden1__type': ['Rectifier'],
        'hidden1__units': [200],
        'hidden2__type': ['Tanh'],
        'hidden2__units': [10]
    }

    clf = SknnClassifier(
        layers=[
            SknnLayer("Rectifier", units=50),
            SknnLayer("Rectifier", units=200),
            SknnLayer("Tanh", units=10),
            SknnLayer("Softmax")],
        learning_rate=0.005,
        n_iter=30)

    x_train, x_val, y_train, y_val = \
        train_test_split(samples.values, labels.values, test_size=0.20, \
                         random_state=np.random.randint(0, 100))

    #	w_train = np.ndarray(x_train.shape[0])
    #	w_train[y_train == 0] = 0.4
    #	w_train[y_train == 1] = 0.6

    print('   running hyper parameter search and training...')
    grid = hyperparameter_search(clf, params, x_train, y_train, x_val, y_val, columns)

    return grid


def main():
    mitdb_dir = 'data/raw_data'
    stage_dir = 'data/stage_data'
    hdf_dir = 'data/hdf5'
    hdf_filename = 'data/hdf5/mit-bih.hdf'

    try:
        os.mkdir('data')
    except:
        pass
    # setup_data_directories(mitdb_dir, stage_dir, hdf_dir)
    '''

    df = load_all_data(mitdb_dir, stage_dir, hdf_dir, hdf_filename)

    # create two sets of features to train on
    print('Staging data for Machine Learning...')
    samples = extract_and_stage_ml(df,anEqualSampling=True, useCached=False)
    maxVarFeatures, labels, columns = isolate_max_var_features(samples)
    milivoltFeatures, labels, columns = isolate_millivolt_features(samples)

    print('Executing Random Forest on raw milivolts....')
    exampleRandomForest(milivoltFeatures, labels, columns)
    '''
    df = load_all_data('mitdb', mitdb_dir, stage_dir, hdf_dir, hdf_filename)

    download_physionet_files('mitdb', mitdb_dir)
    convert_physionet_data_to_csv('mitdb', mitdb_dir, stage_dir)
    build_hdf5_data_store(stage_dir, hdf_filename)
    df = pd.HDFStore(hdf_filename)

    samples = extract_and_stage_ml(df, anEqualSampling=True, useCached=True)
    print('isolate_max_var_features....')
    maxVarFeatures, labels, columns = isolate_max_var_features(samples)
    print('isolate_millivolt_features....')
    milivoltFeatures, labels, columns = isolate_millivolt_features(samples)

    '''
    print('Executing Random Forest on max,var....')
    exampleRandomForest(maxVarFeatures, labels, columns)

    print('Executing Deep Learning on max,var....')
    grid = exampleNeuralNet(maxVarFeatures, labels, columns)
    print(grid.best_params_)


    print('Executing KernelDensity on raw milivolts....')
    grid = exampleKernelDensity(milivoltFeatures, labels, columns)
    '''

    print('Executing Deep Learning on raw milivolts....')
    exampleNeuralNetTfDnn(milivoltFeatures, labels)

    grid = exampleNeuralNet(milivoltFeatures, labels, columns)
    print(grid.best_params_)


__all__ = [
    'load_all_data',
    'extract_and_stage_ml',
    'exampleRandomForest',
    'exampleNeuralNet'
]

if __name__ == '__main__':
    main()
