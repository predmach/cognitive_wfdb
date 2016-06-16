#! /usr/bin/env python
# Charles Pace

# Copyright (c) 2016 Predictive Machines, LLC

'''
 Utility functions to fetch MIT-BIH samples with Annotations
'''
import os
import sys
import re
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
from wfdb import *
from features import *
import subprocess
import StringIO
import csv

def fetchSampleDataWithAnnotations(aSourceDataDirectory, dbname, starttime, endtime):
    scriptPath = os.path.realpath(os.path.dirname(aSourceDataDirectory))
    os.chdir(scriptPath)
    dbname = dbname.replace('mitdb/','')
    stageDir = "data/stage_data"
    allRawDataFiles = os.listdir(stageDir)
    sampleFiles = filter(lambda x: True if re.search(dbname+'.csv$', x) else False,allRawDataFiles)
    if not sampleFiles:
        return ''
    sampleFiles = [os.path.join(stageDir, f) for f in sampleFiles]
    sampleSetNames = [re.search('(\d\d\d)', f).group(1) for f in sampleFiles]
    annotationFiles = filter(lambda x: True if re.search('ann.'+dbname, x) else False,allRawDataFiles)
    annotationFiles = [os.path.join(stageDir, f) for f in annotationFiles]
    #args = ['rdsamp','-r','data/raw_data/200','-f','5:0','-t','5:30','-p','-v']
    args = 'rdsamp -r data/raw_data/' + dbname+ ' -f '+ starttime +' -t '+endtime+' -p -v'
    #print(args)
    sampleData = os.popen(args).read()
    if not sampleData:
        return ' '
    #print("Sample Print")
    args = 'rdann -r data/raw_data/' + dbname+ ' -a atr -f '+ starttime +' -t '+endtime+' -v -x'
    annotationData = os.popen(args).read()
    sf = StringIO.StringIO(sampleData)
    #print("Annnotation::::")
    #print(annotationData)
    sampleDataFrame = pd.read_csv(sf,sep='\t',skiprows=2,skipinitialspace=True,engine="python")
    sampleDataFrame.columns = ['Elapsed time', 'MLII','V1']
    #sampleDataFrame.reset_index(drop=True, inplace=True)
    #sampaf = leDataFrame = sampleDataFrame.ix[1:] # or, skip above
    #sampleDataFrame.reset_index(drop=True, inplace=True)
    #sampleDataFrame.index = sampleDataFrame[1].apply(lambda x: x)
    #print(sampleDataFrame.to_string())
    af = StringIO.StringIO(annotationData)
    correctedStr = parseStringAndAddDelimiter(af)  #Parse String and add delimiter(,) to avoid any parsing issue in pandas
    af = StringIO.StringIO(correctedStr)
    #'\s\s+|\t|\s'
    annotationDataFrame = pd.read_csv(af,sep=',',skiprows=1, engine="python")
    columns = [1,2,4,5,6]
    if(len(annotationDataFrame.columns) > 7):
        columns.append(7)
    annotationDataFrame.drop(annotationDataFrame.columns[columns], axis=1,inplace=True)
    annotationDataFrame.columns = ['Elapsed time', 'Type']
    #print(annotationDataFrame.to_string())
    joined = sampleDataFrame.merge(annotationDataFrame,how='left', on='Elapsed time')
    #joined.to_csv('/home/ubuntu/joined.csv',sep='\t',index=False)
    #return joined.to_string(index=False,sep='\t')
    str = StringIO.StringIO()
    joined.to_csv(str, sep='\t', index=False)
    return str.getvalue()

def parseStringAndAddDelimiter(af):

    lines = af.readlines()
    header = []
    correctedFile = ''
    for index in range(len(lines)):
        if index == 0:
            header = parseColumnInHeader(lines[index])
            correctedFile = correctedFile + (",".join(header))
        else:
            columnData = parseDataColumn(lines[index], len(header))
            correctedFile = correctedFile + (",".join(columnData))
        if index + 1 < len(lines):
            correctedFile = correctedFile + ("\n")

    return correctedFile


#colun name should not have space or tab otherwise it will be considerded as separate column
def parseColumnInHeader(header):
    header = header.replace("\t", " ")
    columns = header.split(" ")
    headerData = []
    for column in columns:
        column = column.strip()
        if len(column):
            headerData.append(column)
    return headerData

def parseDataColumn(strRow, columnCount):
    strRow = strRow.replace("\t", " ")
    columns = strRow.split(" ")
    columnData = []
    for column in columns:
        column = column.strip()
        if len(column):
            columnData.append(column)

    for index in range(columnCount - len(columnData)):
        columnData.append(" ")

    return columnData

def main(args):
    if(len(args) < 3):
        print("Usage:MITDataUtility.py <dbname> <starttime> <endtime>")
        return
    #data = fetchSampleDataWithAnnotations('data/stage_data', '200', '5:00', '5:10')
    data = fetchSampleDataWithAnnotations('/home/ubuntu/cognitive_wfdb/', args[0], args[1], args[2])
    print(data)
'''
Note: Run from root directory
'''
if __name__ == '__main__':
    main(sys.argv[1:])
