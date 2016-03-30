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
from StringIO import StringIO
import re
import os

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from sknn.platform import gpu32
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sknn.mlp import Classifier as SknnClassifier
from sknn.mlp import Layer as SknnLayer
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from wfdb import *
import mitdb
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
	    'n_estimators':      	[11],
	    'n_jobs':              	[-1]
	}

	clf = RandomForestClassifier()# Let's run RandForest -> verify that 
	x_train, x_val, y_train, y_val = \
	    train_test_split(samples, labels, test_size=0.2,
	                     random_state=np.random.randint(0, 100))
	    
	print('   running hyper parameter search and training...')
	grid = hyperparameter_search(clf, params, x_train, y_train, x_val, y_val, columns)

	return grid


def exampleNeuralNet(samples, labels, columns):
	# loads it all to start (do once):
	#load_all_data('data/raw_data', 'data/stage_data', 'data/hdf5', 'data/hdf5/mit-bih.hdf')

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

	params={
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
	grid = hyperparameter_search(clf, params, x_train, y_train, x_val, y_val, columns )

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
	'''
	# setup_data_directories(mitdb_dir, stage_dir, hdf_dir)
	df = load_all_data(mitdb_dir, stage_dir, hdf_dir, hdf_filename)

	# create two sets of features to train on
	print('Staging data for Machine Learning...')
	samples = extract_and_stage_ml(df,anEqualSampling=True, useCached=False)
	maxVarFeatures, labels, columns = isolate_max_var_features(samples)
	milivoltFeatures, labels, columns = isolate_millivolt_features(samples)

	print('Executing Random Forest on raw milivolts....')
	exampleRandomForest(milivoltFeatures, labels, columns)
	'''
	df = mitdb.load_all_data(mitdb_dir, stage_dir, hdf_dir, hdf_filename)
	'''
	download_physionet_files('mitdb',mitdb_dir)
	convert_physionet_data_to_csv('mitdb',mitdb_dir, stage_dir)
 	mitdb.build_hdf5_data_store(stage_dir, hdf_filename)
 	df = pd.HDFStore(hdf_filename)
 	'''
	samples = extract_and_stage_ml(df,anEqualSampling=True, useCached=True)
	print('isolate_max_var_features....')
	maxVarFeatures, labels, columns = isolate_max_var_features(samples)
	print('isolate_millivolt_features....')
	milivoltFeatures, labels, columns = isolate_millivolt_features(samples)

	'''
	print('Executing Random Forest on max,var....')
	exampleRandomForest(maxVarFeatures, labels, columns)

	print('Executing Deep Learning on max,var....')
	exampleNeuralNet(maxVarFeatures, labels, columns)


	print('Executing KernelDensity on raw milivolts....')
	grid = exampleKernelDensity(milivoltFeatures, labels, columns)
	'''

	print('Executing Deep Learning on raw milivolts....')
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