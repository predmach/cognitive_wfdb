#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	modelling - ECG data processing, analysis, validation, and visualization

	generate_all_sample_record_intervals	- generate features for ML
	hyperparameter_search 					- grid search the hyperparameter space
	search_grid								- more than grid search...


'''

from __future__ import print_function, with_statement
from StringIO import StringIO
import re

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

from wfdb import *
from analysis import *


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
	ecgFilter = filter(lambda x: re.search('ECG_Record_', x), anEcgDataFrame.keys())
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


def hyperparameter_search(clf, params, x_train, y_train, x_val, y_val, cols,cv=5):
	'''
		prints results of GridSearch training

		Args:
			clf 		: sklearn classifier to use
			params 		: GridSearch params to use
			x_train 	: x training data
			y_train 	: y training labels
			x_val 		: x validation data
			y_val 		: y validation labels
			cols 		: column names

		Returns:
			grid 
	'''
	# Training
	grid = GridSearchCV(clf, params, cv=cv)
	grid.fit(x_train, y_train)

	# Validation	
	pred = grid.best_estimator_.predict(x_val)
	print(classification_report(y_val, pred))
	pred = grid.best_estimator_.predict(x_train)
	print(classification_report(y_train, pred))

	if isinstance(clf, RandomForestClassifier): 
		importances = grid.best_estimator_.feature_importances_
		print(Series(importances, index=cols).sort_values(ascending=False).head(30))
	return grid

def search_grid(clf, params, x_train, y_train, cols, testSizeRange=np.arange(0.90, 0.995, 0.02)):
	'''
		runs multiple gridsearches on increasingly smaller train sizes and aggregaates the results

		Args:
			clf     	: sklearn classifier to use
			params 		: GridSearch params to use
			x_train 	: x training data
			y_train 	: y training labels
			cols		: column names
			testSizeRange

		Returns:
			reports (list of classification report DataFrames)
			importances (list of importances Series')
	'''
	reports = []
	importances = []
	for test_size in testSizeRange:
		x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=np.random.randint(0, 100))
	
		# Training
		grid = GridSearchCV(clf, params, cv=2)
		grid.fit(x_train_, y_train_)

		# Validation
		pred = grid.best_estimator_.predict(x_val)
		report = classification_report(y_val, pred)
		report = pd.read_table(StringIO(re.sub('avg / total', 'avg/total' , report)), sep=' +', engine='python')
		report['train_size'] = x_train_.shape[0]
		reports.append(report)
		if isinstance(clf, RandomForestClassifier): 
			imp = grid.best_estimator_.feature_importances_
			imp = Series(imp, index=cols).sort_values(ascending=False)
			importances.append(imp)
			
	return reports, importances

def main():
	pass

__all__ = [
	'generate_all_sample_record_intervals',
	'hyperparameter_search',
	'search_grid'
]

if __name__ == '__main__':
	help(main)