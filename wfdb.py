#! /usr/bin/env python
# Charles Pace 

# Copyright (c) 2016 Predictive Machines, LLC

'''
	wfdb - data extraction and manipulation from Physionet MIT-BIH ECG database

	has_words						- string search
	download_annotation_metadata	- extract metadata from physionet - class labels
	download_physionet_mitdb		- run wfdb utilities to extract physionet mitdb data
	convert_mitdb_data_to_csv		- translate the mitdb data into CSV & TXT
	import_and_combine_samples_with_annotations	- merge transformed data
	import_sample_data_into_data_frame			- load and transform CSV sample data
	import_annotation_file_into_data_frame		- load and transform TXT annotation data
	build_mitdb_hdf5_data_store					- driver for transforms & load HDF5
	record_needs_update							- determine if incremental processing can happen
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


def download_physionet_mitdb( aTargetDataDirectory, shouldClean=False ):
	'''
	Download MIT-BIH  data files from Physionet

	Args:
		aTargetDataDirectory : directory in which to store raw files

	Returns:
		None
	'''

	urlMitDB = 'https://www.physionet.org/physiobank/database/mitdb/'
	htmlMitDB = requests.get(urlMitDB).content

	# Scrape the list of all data files out of MIT-BIH page
	scraper = BeautifulSoup(htmlMitDB, "lxml")
	hrefElements = [pageElement['href'] for pageElement in scraper.find_all('a',href=True)]
	dataElements = filter(lambda pageElement: re.search('\.atr|\.dat|\.hea', pageElement, re.I), hrefElements)
	dataElements = sorted(dataElements)
	downloadURLList = [urlMitDB + dataLink for dataLink in dataElements]
	
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



def convert_mitdb_data_to_csv( aSourceDirectory, aTargetDirectory, shouldClean=False):
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
	targetSampleFile = aTargetDirectory + '/mitdb.{name}.csv'
	convertSamples = 'rdsamp -r mitdb/{name} -c -v -pe > {stdout}'
	targetAnnotationFile = aTargetDirectory + '/mitdb.ann.{name}.txt'
	convertAnnotation = 'rdann -r mitdb/{name} -a atr -v -e > {stdout}'

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
	# load samples and annotations into a single frame
	sampleDataFrame = import_sample_data_into_data_frame(aSampleCsvFile)
	annotationDataFrame = import_annotation_file_into_data_frame(anAnnotationTxtFile)
	combinedDataFrame = pd.concat([sampleDataFrame,annotationDataFrame], axis=1)
	
	# Dowmload labels from MIT-BIH site
	arrythmiaSymbols = aMetadataSet[aMetadataSet.arrythmia].Symbol.tolist()
	#print(annotationSymbols)

	''' Pull "arrythmioa events" out of data
	'''
	#  Convert Type and Aux to integer values
	arrythmiaEvents = combinedDataFrame.Type.apply(lambda s: s in arrythmiaSymbols).astype(int)
	arrythmiaEvents += combinedDataFrame.Aux.apply(lambda s: s in arrythmiaSymbols).astype(int)

	arrythmiaEvents = arrythmiaEvents.astype(bool).astype(int)
	combinedDataFrame['arrythmia_events'] = arrythmiaEvents
	
	# limit to normal events
	normalSymbols = ['N', 'L', 'R']
	normalEvents = combinedDataFrame.Type.apply(lambda s: s in normalSymbols).astype(int)
	normalEvents += combinedDataFrame.Aux.apply(lambda s: s in normalSymbols).astype(int)

	normalEvents = normalEvents.astype(bool).astype(int)
	combinedDataFrame['normal_events'] = normalEvents
	
	normalIndex = combinedDataFrame[combinedDataFrame.normal_events == 1].index
	arrythmiaIndex = combinedDataFrame[combinedDataFrame.arrythmia_events == 1].index
	print( '{0} arrythmia events, {1} normal events'.format(len(arrythmiaIndex),len(normalIndex)))

	return combinedDataFrame

def build_mitdb_hdf5_data_store(aSourceDataDirectory, aTargetHDF5):
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
	annotationMetadata.to_hdf(aTargetHDF5, 'ECG_Annotation_Metadata')

	for sampleSet, sampleFile, annotationFile in zip(sampleSetNames, sampleFiles, annotationFiles):

		recordName = 'ECG_Record_' + sampleSet

		# skip if we've already imported this data
		if not record_needs_update(recordName, sampleFile, annotationFile, aTargetHDF5):
			continue

		print( recordName )
		combinedDataFrame = import_and_combine_samples_with_annotations(sampleFile, annotationFile, annotationMetadata)
		combinedDataFrame.to_hdf(aTargetHDF5, recordName)

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
			record = pd.read_hdf(aTargetHDF5, 'ECG_import_checksums' )
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

	record.to_hdf(aTargetHDF5, 'ECG_import_checksums')

	return True


def main():
	pass

__all__ = [
	'has_words',
	'download_annotation_metadata',
	'download_physionet_mitdb',
	'convert_mitdb_data_to_csv',
	'import_and_combine_samples_with_annotations',
	'import_sample_data_into_data_frame',
	'import_annotation_file_into_data_frame',
	'build_mitdb_hdf5_data_store',
	'record_needs_update'
]

if __name__ == '__main__':
	help(main)