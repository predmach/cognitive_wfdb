#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	mitdb - data extraction and manipulation from Physionet MIT-BIH ECG database

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


def load_all_data(aMitDbDir, aStagingDir, anHdfDir, anHdfFilename):
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

	download_physionet_files('mitdb',aMitDbDir)
	convert_physionet_data_to_csv('mitdb',aMitDbDir, aStagingDir)
 	build_hdf5_data_store(aStagingDir, anHdfFilename)

	ecgDataframe = pd.HDFStore(anHdfFilename)
	return ecgDataframe

def download_annotation_metadata():
	'''
	

	Args:
		None

	Returns:
		DataFrame
	'''

	# Scrape annotation table frm MIT_-BIH info page
	annotationURL = 'https://www.physionet.org/physiobank/database/html/mitdbdir/intro.htm#annotations'
	htmlAnnotations = requests.get(annotationURL).content
	scraper = BeautifulSoup(htmlAnnotations, "lxml")
	tableElements = scraper.select('table')[-1]

	# 
	metadata = pd.read_html(str(tableElements), header=0)[0]
	mask = metadata.Symbol.apply(lambda x: True) 
	mask.ix[[20,36]] = False
	metadata = metadata[mask]

	metadata.reset_index(drop=True, inplace=True)
	metadata.loc[0, 'Symbol'] = 'N'
	metadata.loc[37, 'Symbol'] = 'M'
	metadata.loc[38, 'Symbol'] = 'P'
	metadata.loc[39, 'Symbol'] = 'T'

	
	lut = {
		'artifact':  ['artifact', 'Unclassified', 'Non-conducted', 'Fusion'],
		'arrythmia': ['flutter', 'bigeminy', 'tachycardia', 'fibrillation'],
		'other':     ['bradycardia', 'Abberated', 'Premature', 'escape'],
		'signal':    ['Signal quality', 'Extreme noise', 'Missed beat', 'Pause', 'Tape slippage']
	}

	for i in lut.keys():	
		metadata[i] = metadata.Meaning.apply(lambda x: has_words(x, lut[i]))

	return metadata

def import_sample_data_into_data_frame(aSampleFile):
	'''
	

	Args:
		aSampleFile :  CSV MIT_BIH data

	Returns:
		Pandas data frame
	'''

	dataframe = pd.read_csv(aSampleFile, skiprows=2) # , encoding='utf-16', header=None)
	dataframe.columns = ['Elapsed_Microseconds', 'MLII_milliVolts', 'V5_milliVolts']
	dataframe.reset_index(drop=True, inplace=True)
	dataframe = dataframe.ix[1:] # or, skip above
	dataframe.reset_index(drop=True, inplace=True)

	# Set data types	
	dataframe.MLII_milliVolts = dataframe.MLII_milliVolts.astype(float)
	dataframe.V5_milliVolts = dataframe.V5_milliVolts.astype(float)
	
	# Change the time to a zero base and apply as the index of the data frame
	baseTime = datetime.strptime('00:00.00', '%M:%S.%f')
	dataframe.index = dataframe.Elapsed_Microseconds.apply(lambda x: datetime.strptime(x[1:-1], '%M:%S.%f') - baseTime)
	dataframe.drop('Elapsed_Microseconds', axis=1, inplace=True)

	return dataframe

def import_annotation_file_into_data_frame(anAnnotationFile):
	'''
	
	Args:
		anAnnotationFile : 

	Returns:
		DataFrame
	'''
	dataframe = pd.read_table(anAnnotationFile, sep='\s\s+|\t| C')
	dataframe.columns = ['Elapsed_Microseconds', 'Sample_num', 'Type', 'Sub', 'Chan', 'Num', 'Aux']

	# Change the time to a zero base and apply as the index of the data frame
	baseTime = datetime.strptime('00:00.00', '%M:%S.%f')
	dataframe.index = dataframe.Elapsed_Microseconds.apply(lambda x: datetime.strptime(x, '%M:%S.%f') - baseTime)
	dataframe.drop('Elapsed_Microseconds', axis=1, inplace=True)
	
	dataframe.Sample_num = dataframe.Sample_num.astype(int)
	dataframe.Sub = dataframe.Sub.astype(int)
	dataframe.Chan = dataframe.Chan.astype(int)
	dataframe.Num = dataframe.Num.astype(int)
	# Type, Aux, 
	
	return dataframe

def import_and_combine_samples_with_annotations(aSampleCsvFile, anAnnotationTxtFile, aMetadataSet):
	'''
	

	Args:
		aSampleCsvFile : MIT-BIH file
		anAnnotationTxtFile : 

	Returns:
		Pandas data frame
	'''
	# load samples and annotations, then merge them using timestamp into a single frame
	sampleDataFrame = import_sample_data_into_data_frame(aSampleCsvFile)
	annotationDataFrame = import_annotation_file_into_data_frame(anAnnotationTxtFile)
	df = pd.concat([sampleDataFrame,annotationDataFrame], axis=1)
	
	#  Convert Type and Aux to integer values
	# Labels from MIT-BIH site
	arrythmiaSymbols = aMetadataSet[aMetadataSet.arrythmia].Symbol.tolist()
	normalSymbols = ['N', 'L', 'R']

	labels = ['arrythmia','normal']
	symbolSets = [arrythmiaSymbols,normalSymbols]
	annotationList = ['Type','Aux']

	for label, symbols in zip(labels,symbolSets):

		eventName = label+'_events'
		df[eventName] = 0

		for annotation in annotationList:
			df[eventName] = df.apply( lambda x: 1 if x[annotation] in symbols else x[eventName], axis=1)

	# add up each, just to see...
	print('calculating event occurances...')
	for label in labels:
		numEvents = len(df[df[label+'_events'] == 1].index)
		print( '{0} {1} events'.format(numEvents, label))

	return df

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
		combinedDataFrame = import_and_combine_samples_with_annotations(sampleFile, annotationFile, annotationMetadata)
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

def main():
	pass

__all__ = [
	'load_all_data',
	'download_annotation_metadata',
	'import_and_combine_samples_with_annotations',
	'import_sample_data_into_data_frame',
	'import_annotation_file_into_data_frame',
	'build_hdf5_data_store',
	'generate_normal_and_arrythmia_samples'
]

if __name__ == '__main__':
	help(main)