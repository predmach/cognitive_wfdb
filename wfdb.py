#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	wfdb - data extraction and manipulation from Physionet MIT-BIH ECG database

	has_words						- string search
	download_annotation_metadata	- extract metadata from physionet - class labels
	download_physionet_files		- run wfdb utilities to extract physionet mitdb data
	convert_data_to_csv				- translate the mitdb data into CSV & TXT
	record_needs_update				- determine if incremental processing can happen
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
	
#!! CHANGE THIS FOR SUBSTRING SEARCH
def has_words(aTestString, aKeyList):

	for keyString in aKeyList:
		if not aTestString.find(keyString) == -1:
		# if keyString.lower() in aTestString.lower():
			return True
	return False




def download_physionet_files( aDatabase='mghdb', aTargetDataDirectory='./data', shouldClean=False, useExtensions=['atr','dat','hea'] ):
	'''
	Download MGH/MF data files from Physionet

	Args:
		aTargetDataDirectory : directory in which to store raw files
		aDatabase		: default 'mghdb'
		shouldClean		: reload the data?
		useExtensions	: different databases will have different files ['atr','dat','hea']

	Returns:
		None
	'''

	extensionRegex = ''
	for ext in useExtensions:
		if len(extensionRegex) > 1:
			extensionRegex += '|'
		extensionRegex = extensionRegex + '\.' + ext

	urlPhysionetDB = 'https://www.physionet.org/physiobank/database/' + aDatabase + '/'
	htmlDB = requests.get(urlPhysionetDB).content

	# Scrape the list of all data files out of database page
	scraper = BeautifulSoup(htmlDB, "lxml")
	hrefElements = [pageElement['href'] for pageElement in scraper.find_all('a',href=True)]
	dataElements = filter(lambda pageElement: re.search(extensionRegex, pageElement, re.I), hrefElements)
	dataElements = sorted(dataElements)
	downloadURLList = [urlPhysionetDB + dataLink for dataLink in dataElements]
	
	targetFileList = [os.path.join(aTargetDataDirectory, fileName) for fileName in dataElements]
	i = 0
	for dataURL, localDataFile in zip(downloadURLList, targetFileList):
		i += 1
#		if i > 10*3:
#			break

		if os.path.isfile(localDataFile) and not shouldClean : # don't re-download
			continue

		print('downloading {0}'.format(localDataFile))
		with open(localDataFile, 'w+') as localFileHandle:
			localFileHandle.write(requests.get(dataURL).content)

def convert_physionet_data_to_csv( aDatabase, aSourceDirectory, aTargetDirectory, shouldClean=False):
	'''
	Convert raw data to CSV & TXT

	Args:
		aSourceDirectory : 
		aTargetDirectory : 

	Returns:
		None
	'''

	rawDataFiles = set()
	for dataFileName in os.listdir(aSourceDirectory):
		found = re.search('^(\d\d\d)\.', dataFileName)
		if found:
			rawDataFiles.add(found.group(1))

	conversionProcesses = set()
	targetSampleFile = aTargetDirectory + '/' + aDatabase + '.{name}.csv'
	convertSamples = 'rdsamp -r '+ aSourceDirectory +'/{name} -c -v -pe > {stdout}'
	targetAnnotationFile = aTargetDirectory + '/' + aDatabase + '.ann.{name}.txt'
	convertAnnotation = 'rdann -r '+ aSourceDirectory +'/{name} -a atr -v -e > {stdout}'

	maxOpenFiles = 6
	lowOpenFiles = 3
	numOpenFiles = 0

	for rawDataFile in rawDataFiles:
        
		targetSample = targetSampleFile.format(name=rawDataFile)
		if not os.path.isfile(targetSample) and not shouldClean:
			print(targetSample)
			sampleProcess = Popen(convertSamples.format(name=rawDataFile,stdout=targetSample), shell=True, stdout=PIPE)
			conversionProcesses.add( sampleProcess )
			numOpenFiles += 1

		targetAnnotation = targetAnnotationFile.format(name=rawDataFile)
		if not os.path.isfile(targetAnnotation) and not shouldClean:
			annotationProcess = Popen(convertAnnotation.format(name=rawDataFile,stdout=targetAnnotation), shell=True, stdout=PIPE)
			conversionProcesses.add( annotationProcess )
			numOpenFiles += 1

		if numOpenFiles > maxOpenFiles:
			print( 'Reached max processes {0} pending'.format(numOpenFiles) )
			for conversionProcess in conversionProcesses:
				numOpenFiles -= 1
				if numOpenFiles == lowOpenFiles:
					break
				conversionProcess.communicate()
			conversionProcesses.clear()
			numOpenFiles = 0


	for conversionProcess in conversionProcesses:
		conversionProcess.communicate()
 

def record_needs_update(aRecordName, aSampleFile, anAnnotationFile, aTargetHDF5):
	'''
	use an import data frame to keep track of the timestamps on the source files,

	Args:


	Returns:
		None
	'''
	checksum = '[{0}{1}]'.format(time.ctime(os.path.getmtime(aSampleFile)),time.ctime(os.path.getmtime(anAnnotationFile)))
	try:
		try:
			record = pd.read_hdf(aTargetHDF5, 'Import_checksums' )
		except KeyError as e:
			record = pd.DataFrame({'record_name':aRecordName,'checksum':checksum},index=[0])
			pass
		else:
			checksum_ = record[record['record_name']==aRecordName].checksum.tolist()[0]
			if checksum == checksum_:
				return False

	except IndexError:
		record = record.append({'record_name':aRecordName,'checksum':checksum},ignore_index=True)
	else:
		record.ix[record.record_name==aRecordName, 'checksum'] = checksum

	record.to_hdf(aTargetHDF5, 'Import_checksums')

	return True

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
	'has_words',
	'download_physionet_files',
	'convert_physionet_data_to_csv',
	'record_needs_update',
	'setup_data_directories',
	'generate_time_interval',
	'generate_sample_intervals',
	'max_amplitude_filter'
]

if __name__ == '__main__':
	help(main)