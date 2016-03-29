#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	analysis - ECG data processing, analysis, validation, and visualization

	setup_data_directories				- delete/create data directory structure
	raw_data_staging_and_persistance	- extract data from physionet
	generate_time_interval				- generate features for ML
	generate_sample_intervals			- 
	max_amplitude_filter				- convolve signals with moments & kurtosis
	generate_normal_and_arrythmia_samples - utility function

'''

from __future__ import print_function, division #, with_statement

# Base
import os
import re
import itertools

#M Multi-proc
from subprocess import Popen, PIPE
from datetime import datetime

# Algorithmic
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

import shutil

from wfdb import *


def raw_data_staging_and_persistance(aRawDirectory, aStageDirectory, aDataStoreName):
	'''

	Args:
		aRawDirectory : where to store the downloaded data
		aStageDirectory : where to convert data to CSV & TXT
		aDataStoreName : where to establish the HDF5 DataStore

	Returns:
		none
	'''
	download_physionet_mitdb(aRawDirectory)

	convert_mitdb_data_to_csv(aRawDirectory, aStageDirectory)

 	build_mitdb_hdf5_data_store(aStageDirectory, aDataStoreName)



def setup_data_directories(aRawDirectory, aStageDirectory, aDataStoreName):
	'''

	Args:
		aRawDirectory : where to store the downloaded data
		aStageDirectory : where to convert data to CSV & TXT
		aDataStoreName : where to establish the HDF5 DataStore

	Returns:
		none
	'''
	#shutil.rmtree(aRawDirectory,True)
	#shutil.rmtree(aStageDirectory,True)
	#shutil.rmtree(aDataStoreName,True)
	os.mkdir(aRawDirectory)
	os.mkdir(aStageDirectory)
	os.mkdir(aDataStoreName)

def generate_time_interval(aFormatString=None, hours=0, minutes=0, seconds=0, microseconds=0):
	'''

	Args:
		aFormatString 
		hours 			: number of hours
		minutes 		: number of minutes
		seconds 		: number of seconds
		mmicroseconds 	: number of mmicroseconds

	Returns:
		time intevrval
	'''
	if not aFormatString:
		aFormatString = ':'.join(map(str, [hours, minutes, seconds])) + '.' + str(microseconds)

	return datetime.strptime(aFormatString, '%H:%M:%S.%f') - datetime.strptime('0:0:0.0', '%H:%M:%S.%f')

def max_amplitude_filter(aSingleChannel):
	'''
	

	Args:
		aSingleChannel  :	 mitdb DataFrame

	Returns:
		heartbeats (list): temporal list of heartbeat probabilities
	'''
	#x1 = aDataFrame.index.astype(int).tolist()

	modeFit = pd.rolling_kurt(aSingleChannel, 100)
	stdDev = pd.rolling_std(aSingleChannel - pd.rolling_mean(aSingleChannel, 10), 10)
	return aSingleChannel * modeFit * stdDev
#	return reduce( lambda x,y: x*y, [y1, modeFit, y3] )

def generate_sample_intervals(aDataFrame, aTimeIndex, aLabel, aStartInterval, anEndInterval):
	'''
	

	Args:
		aDataFrame : mitdb data
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
		intervalSamples = aDataFrame.loc[start:end, ['MLII_milliVolts', 'V5_milliVolts']]
		intervalSeries = Series(intervalSamples.as_matrix().ravel())

		# enhance features in each lead series
		# save off the max and var
		
		lead_MLII_filtered = max_amplitude_filter(intervalSamples['MLII_milliVolts'])
		intervalSeries['MLII_peak_max'] = lead_MLII_filtered.max()
		intervalSeries['MLII_peak_var'] = lead_MLII_filtered.var()
		
		lead_v5_filtered = max_amplitude_filter(intervalSamples['V5_milliVolts'])
		intervalSeries['V5_peak_max'] = lead_v5_filtered.max()
		intervalSeries['V5_peak_var'] = lead_v5_filtered.var()
				
		intervalSeries['labels'] = aLabel
		
		sampleIntervals.append(intervalSeries)
	
	return DataFrame(sampleIntervals)

#def generate_normal_and_arrythmia_samples(anECGDataFrame,anOffsetWindow=[500000,10000]):
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

	normalFeatures = generate_sample_intervals(anECGDataFrame,normalIndex, 0, startInterval, endInterval)
	arrythmiaFeatures = generate_sample_intervals(anECGDataFrame,arrythmiaIndex, 1, startInterval, endInterval)

	# combine data frames
	if arrythmiaFeatures.shape[0] > 0:
		normalFeatures = pd.concat([normalFeatures, arrythmiaFeatures])

	return normalFeatures



	def main():
		pass
# ------------------------------------------------------------------------------

__all__ = [
	'setup_data_directories',
	'raw_data_staging_and_persistance',
	'generate_time_interval',
	'generate_sample_intervals',
	'max_amplitude_filter',
	'generate_normal_and_arrythmia_samples'
]

if __name__ == '__main__':
	help(main)