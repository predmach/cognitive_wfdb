#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	features - routines for generating features from Physionet data 

	import_and_combine_samples_with_annotations	- merge transformed data
	import_sample_data_into_data_frame			- load and transform CSV sample data
	import_annotation_file_into_data_frame		- load and transform TXT annotation data
	build_mitdb_hdf5_data_store					- driver for transforms & load HDF5
]
'''

from __future__ import print_function, division #, with_statement

# Base
import os
import re
import time
import itertools

#M Multi-proc
from subprocess import Popen, PIPE
from datetime import datetime

# Web
import requests
from bs4 import BeautifulSoup

# Algorithmic
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

from wfdb import *


def load_all_data(aDbName,aDbDir, aStagingDir, anHdfDir, anHdfFilename):
	'''
	a big loader, all routines are incremental, so you can kill and restart

	Args:
		aMitDbDir 		: raw data goes in this directory
		aStagingDir 	: the raw data is converted to csv & txt here
		anHdfDir 		: where the HDF is stored
		anHdfFilename 	: fullpath (i know...) to HDF5 file

	Returns:
		DataFrame
	'''

	download_physionet_files(aDbName,aDbDir)
	convert_physionet_data_to_csv(aDbName,aDbDir, aStagingDir)
 	build_hdf5_data_store(aStagingDir, anHdfFilename)

	ecgDataframe = pd.HDFStore(anHdfFilename)
	return ecgDataframe


def build_hdf5_data_store(aSourceDataDirectory, aTargetHDF5):
	'''
	writes all mitdb source directory csvs and txt annotations to a single hdf5 file

	Args:
		aSourceDataDirectory: full path of directory containing mitdb csvs and txt annotations
		aTargetHDF5: fullpath of target file

	Returns:
		None
	'''

	allRawDataFiles = os.listdir(aSourceDataDirectory)

	sampleFiles = filter(lambda x: True if re.search('csv$', x) else False, allRawDataFiles)
	sampleFiles = [os.path.join(aSourceDataDirectory, f) for f in sampleFiles]
	sampleFiles = sorted(sampleFiles)

	sampleSetNames = [re.search('(\d\d\d)', f).group(1) for f in sampleFiles]
	
	annotationFiles = filter(lambda x: True if re.search('ann', x) else False, allRawDataFiles)
	annotationFiles = [os.path.join(aSourceDataDirectory, f) for f in annotationFiles]
	annotationFiles = sorted(annotationFiles)
	
	annotationMetadata = download_annotation_metadata()
	annotationMetadata.to_hdf(aTargetHDF5, 'Annotation_Metadata')

	for sampleSet, sampleFile, annotationFile in zip(sampleSetNames, sampleFiles, annotationFiles):

		recordName = 'Record_' + sampleSet

		# skip if we've already imported this data
		if not record_needs_update(recordName, sampleFile, annotationFile, aTargetHDF5):
			continue

		print( recordName )
		combinedDataFrame = import_samples_and_annotations(sampleFile, annotationFile, annotationMetadata)
		combinedDataFrame.to_hdf(aTargetHDF5, recordName)

def generate_normal_and_arrythmia_samples(anECGDataFrame,anOffsetWindow=[5000,5000]):
	'''

	Args:
		anECGDataFrame : mitdb data
		anOffsetWindow

	Returns:
		DataFrame containing features for 
	'''
	
	windowStartOffset = anOffsetWindow[0]
	windowEndOffset = anOffsetWindow[1]
	startInterval = generate_time_interval(microseconds=windowStartOffset)
	endInterval = generate_time_interval(microseconds=windowEndOffset)

	normalIndex = anECGDataFrame[anECGDataFrame.normal_events == 1].index
	arrythmiaIndex = anECGDataFrame[anECGDataFrame.arrythmia_events == 1].index

	leadNames = ['MLII_milliVolts', 'V5_milliVolts']
	normalFeatures = generate_sample_intervals(anECGDataFrame,normalIndex, 0, startInterval, endInterval,leadNames)
	arrythmiaFeatures = generate_sample_intervals(anECGDataFrame,arrythmiaIndex, 1, startInterval, endInterval,leadNames)

	# combine data frames
	if arrythmiaFeatures.shape[0] > 0:
		normalFeatures = pd.concat([normalFeatures, arrythmiaFeatures])

	return normalFeatures



def max_amplitude_filter(aSingleChannel):
	'''
	

	Args:
		aSingleChannel  :	  DataFrame

	Returns:
		filtered (list): temporal list of filtered values, peaks accentuated
	'''

	modeFit = pd.rolling_kurt(aSingleChannel, 100)
	stdDev = pd.rolling_std(aSingleChannel - pd.rolling_mean(aSingleChannel, 10), 10)
	return aSingleChannel * modeFit * stdDev

def generate_sample_intervals(aDataFrame, aTimeIndex, aLabel, aStartInterval, anEndInterval,
		aColumnList):
	'''
	

	Args:
		aDataFrame : time series data
		aTimeIndex : index of events in data
		aLabel : class to associate with these events
		aStartInterval : how far back to go 


	Returns:
		DataFrame
	'''

	# for each event, generate an interval around it
	startIntervalList = aTimeIndex - aStartInterval
	endIntervalList = aTimeIndex + anEndInterval

	intervals = zip(startIntervalList,endIntervalList)
	sampleIntervals = []

	# for each event interval, save off series data and also features
	for start,end in intervals[1:]:

		# all mV values in a single series
		intervalSamples = aDataFrame.loc[start:end, aColumnList]
		intervalSeries = Series(intervalSamples.as_matrix().ravel())

		# enhance features in each lead series
		# save off the max and var
		for signalName in aColumnList:
			lead_filtered = max_amplitude_filter(intervalSamples[signalName])
			intervalSeries[signalName+'_max'] = lead_filtered.max()
			intervalSeries[signalName+'_var'] = lead_filtered.var()
		
		intervalSeries['labels'] = aLabel
		
		sampleIntervals.append(intervalSeries)
	
	return DataFrame(sampleIntervals)

def main():
	pass

__all__ = [
	'load_all_data',
	'build_hdf5_data_store',
	'generate_normal_and_arrythmia_samples',
	'generate_sample_intervals',
	'max_amplitude_filter']

if __name__ == '__main__':
	help(main)