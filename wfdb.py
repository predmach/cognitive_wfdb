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
 

def import_sample_data(aSampleFile):
	'''
	

	Args:
		aSampleFile :  CSV PHYSIONET data

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

def import_annotation_data(anAnnotationFile):
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
	
	return dataframe


def import_samples_and_annotations(aSampleCsvFile, anAnnotationTxtFile, aMetadataSet):
	'''
	

	Args:
		aSampleCsvFile : MIT-BIH file
		anAnnotationTxtFile : 

	Returns:
		Pandas data frame
	'''

	# load samples and annotations, then merge them using timestamp into a single frame
	sampleDataFrame = import_sample_data(aSampleCsvFile)
	annotationDataFrame = import_annotation_data(anAnnotationTxtFile)
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



def main():
	pass

__all__ = [
	'has_words',
	'download_annotation_metadata',
	'download_physionet_files',
	'convert_physionet_data_to_csv',
	'import_sample_data',
	'import_annotation_data',
	'import_samples_and_annotations',
	'record_needs_update',
	'setup_data_directories',
	'generate_time_interval'

]

if __name__ == '__main__':
	help(main)