#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:01:14 2017

@author: AlexandraDing
"""

import numpy as np
import pandas as pd
import imdb
from imdb import IMDb
from datetime import datetime
import pickle
import os
os.chdir('/Users/AlexandraDing/Documents/cs109b-best-group')

movie_df = pickle.load(open("add_imdb", 'rb' ))

### Try to save to CSV
#movie_df.to_csv('test_add_imdb_csv.csv', index = False)
# UnicodeEncodeError: 'ascii' codec can't encode character u'\xf4' in position 2: ordinal not in range(128)

# Try to save each column as a CSV, figure out which ones don't work
junk_directory = '/Users/AlexandraDing/Documents/cs109b-best-group/fix_imdb_encoding/'

error_columns = []
for i in range(len(movie_df.columns)):
    try:
        movie_df[movie_df.columns[i]].to_csv(junk_directory + 'test' + str(i) +'.csv')
    except Exception,e: 
        print str(e) + ' error at ' + str(i)
        error_columns.append(i)
 
#'ascii' codec can't encode character u'\xf1' in position 15: ordinal not in range(128) error at 35
#'ascii' codec can't encode character u'\xf4' in position 2: ordinal not in range(128) error at 37

### examine error columns: 35, 37 + encode all text in utf-8
for bad_col in error_columns:
    movie_df[movie_df.columns[bad_col]] = movie_df[movie_df.columns[bad_col]].str.encode('utf-8')


### Actually save the thing
movie_df.to_csv('add_imdb_utf8.csv', index = False)

try:
    W = pd.read_csv('add_imdb_utf8.csv')
    print 'it worked!'
except:
    print 'nope its still messed up'

