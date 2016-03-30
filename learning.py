#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	learning.py - 

	extract_and_stage_ml	- load from hdf5, convert into features
	generate_all_sample_record_intervals
	isolate_max_var_features				- just max & var for each lead
	isolate_millivolt_features				- larger feature vector

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
from mitdb import *


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

def generate_all_sample_record_intervals(anEcgDataFrame, anEqualSampling=True):
	'''
	grab records from HDS5 Datastore, process into huge set of feature vectors

	Args:
		anEcgDataFrame : mitDB data
		anEqualSampling : make equal number of each class

	Returns:
		DataFrame
	'''

	# get a list of all the recordings
	ecgFilter = filter(lambda x: re.search('Record_', x), anEcgDataFrame.keys())
	ecgDataFrames = [anEcgDataFrame[k] for k in ecgFilter] # replace with equery
	
	# generate all the sample data intervals for each record that has arrythmias
	# each record will have time around each annotated event
	# also, derived features are around each event
	mlStage = DataFrame()
	for record in ecgDataFrames:
		if record.arrythmia_events.sum() > 1:
			if len(record[record.arrythmia_events == 1].index) == 0:
				continue

			recordSamples = generate_normal_and_arrythmia_samples(record)
			mlStage = pd.concat([mlStage, recordSamples ])

	mlStage.reset_index(drop=True, inplace=True)

	if len(mlStage.index) == 0 :
		print("no arrythmia records, nothing to learn...")
		return

	''' Reduce the number of normal samples to match the number of arrithmia samples
	'''
	if anEqualSampling:
		mask = mlStage['labels'] == 1 							# 1 = arrythmia
		size = mlStage[mask].shape[0]							# total arrythmias
		randNormIndex = np.random.choice(mlStage[~mask].index, size) 	# grab random normal
		index = np.concatenate([randNormIndex, mlStage[mask].index])	# 
		mlStage = mlStage.ix[index]
		mlStage.reset_index(drop=True, inplace=True)
		
	return mlStage

def isolate_max_var_features(anEcgDataFrame):
	# trim data down
	data = anEcgDataFrame

	columns = data.drop('labels', axis=1).columns.tolist()
	columns = filter( lambda x: re.search('_max|_var', str(x)), columns )

	samples = data[columns]   
	samples = samples.fillna(0)

	labels = data['labels']

	return samples, labels, columns

def isolate_millivolt_features(anEcgDataFrame):
	# trim data down
	data = anEcgDataFrame

	columns = data.drop('labels', axis=1).columns.tolist()
	columns = filter( lambda x: not re.search('_max|_var', str(x)), columns )

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


def main():
	pass

__all__ = [
	'extract_and_stage_ml',
	'generate_all_sample_record_intervals',
	'isolate_max_var_features',
	'isolate_millivolt_features'
]

if __name__ == '__main__':
	help(main)
