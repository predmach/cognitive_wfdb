#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	learning.py - 

	extract_and_stage_ml	- load from hdf5, convert into features

]
'''

from __future__ import print_function, with_statement
from StringIO import StringIO
import re
import os

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

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
from analysis import *
from modelling import *



def extract_and_stage_ml(anEcgDataFrame, anEqualSampling=True, useCached=True):
	'''
	Query the file for the arrythmia data, bring in normal data too

	Args:
		anEcgDataFrame		: data frame
		anEqualSampling		: equalize the amount of each class

	Returns:
		DataFrame
	'''
	cache = 'cached_eq_ml_data.hdf'
	if useCached and os.path.isfile(cache):
		equalized_data = pd.read_hdf(cache, 'cached_data')
		print('Using cached ML data...')
	else:
		equalized_data = generate_all_sample_record_intervals(anEcgDataFrame)
		if (useCached):
			print('Using cached ML data...')
			equalized_data.to_hdf(cache, 'cached_data')

	return equalized_data

def isolate_max_var_features(anEcgDataFrame):
	# trim data down
	data = anEcgDataFrame

	columns = data.drop('labels', axis=1).columns.tolist()
	columns = filter( lambda x: re.search('peak', str(x)), columns )

	samples = data[columns]   
	samples = samples.fillna(0)

	labels = data['labels']

	return samples, labels, columns

def isolate_millivolt_features(anEcgDataFrame):
	# trim data down
	data = anEcgDataFrame

	columns = data.drop('labels', axis=1).columns.tolist()
	columns = filter( lambda x: not re.search('peak', str(x)), columns )

	samples = data[columns]   
	samples = samples.fillna(0)

	labels = data['labels']

	return samples, labels, columns

'''	ECG convnet
	build up plots 
		- turn each frame into ....something
		create a batch of plots, feed to Inception FineTune
			- grab all data
			- iterate thru batches
			- plot to image
			- group up images for batch
			- execute batch


'''



