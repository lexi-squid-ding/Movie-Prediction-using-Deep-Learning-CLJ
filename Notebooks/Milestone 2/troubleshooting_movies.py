#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 02:36:03 2017

@author: AlexandraDing
"""

#### Checks for malformed entries

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn visualization package
import pandas as pd
import time
import re # Regex
import collections

import tensorflow 
# File reading things
import pickle
import json
import csv
import datetime # For handling dates

# The "requests" library makes working with HTTP requests easier
import requests
import os
from bs4 import BeautifulSoup
#from IPython.display import Image, display # Display image from URL
#from IPython.core.display import HTML # Display image from URL

# TMDB API wrapper
import tmdbsimple as tmdb

#data_wd = '/Users/AlexandraDing/Desktop/cs109b-best-group/'
os.chdir('/Users/AlexandraDing/Documents/cs109b-best-group/Full_Dataset')

movies_list = pickle.load(open('movies_list_2011_2016_pickled.p', 'rb'))



# data_2011_16 = pd.read_csv('combined_data_2011to2016_onehot_genres.csv', header = 0)
# data_2011_16 = pd.read_csv('combined_data_2011to2016_onehot_genres.csv')

# Troubleshooting:
# http://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err
with open('combined_data_2011to2016_onehot_genres.csv', 'rb') as f:
    reader = csv.reader(f)

    linenumber = 1
    try:
        for row in reader:
            linenumber += 1
    except Exception as e:
        print (("Error line %d: %s %s" % (linenumber, str(type(e)), e.message)))

## Attempt to read CSV line by line and skip exceptions
#squid = []
#with open('combined_data_2011to2016_onehot_genres.csv', 'rb') as f:
#    reader = csv.reader(f)
#
#    linenumber = 1
#    for linenumber in range(8000):
#        try:
#            for row in reader:
#                squid.append(reader.next())
#        except Exception as e:
#            print (("Error line %d: %s %s" % (linenumber, str(type(e)), e.message)))
#        
## Delete the offending line and try again?
#linenumber



file_name = 'all_movies_from_2011_2016_data_excl_err.csv'
#file_name = 'all_movies_from_2011_2016_data.csv'

errs = 0
### Save json contents as CSV
with open(file_name, "w") as file:
    csv_file = csv.writer(file)  
    # Add column names
    csv_file.writerow(['poster_path', 'title', 'release_date', 'overview', 'popularity', 'original_title', 'backdrop_path',
                       'vote_count', 'video', 'adult', 'vote_average', 'original_language', 'id', 'genre_ids'])
    # For each item in list, get attributes of movie
    for i in range(len(movies_list)):
        for item in movies_list[i]['results']:

             try:
                 csv_file.writerow([item['poster_path'], item['title'], item['release_date'], item['overview'], 
                                    item['popularity'], item['original_title'], item['backdrop_path'], 
                                    item['vote_count'], item['video'], item['adult'], item['vote_average'], 
                                    item['original_language'], item['id'], item['genre_ids']])
             except:
                 print item

#            csv_file.writerow([item['poster_path'], item['title'].encode('utf8', 'ignore'), item['release_date'], item['overview'].encode("utf8", "ignore"), 
#                   item['popularity'], item['original_title'].encode("utf8", 'ignore'), item['backdrop_path'], 
#                   item['vote_count'], item['video'], item['adult'], item['vote_average'], 
#                   item['original_language'], item['id'], item['genre_ids']])

### Try cleaning CSV directly: get rid of offending entries

# Load data from 2012- 2016
movies_data = pd.read_csv('all_movies_from_2011_2016_data.csv')
movies_data.shape
print movies_data.head(4)

# How to search for string in list
old_list = ['abc123', 'def456\t', 'ghi789']
[old_list[i] for i in range(len(old_list)) if re.search('\t', old_list[i]) ]
 
# Find index of bad entry
[i for i in range(len(old_list)) if re.search('\t', old_list[i]) ]

#[i for i in range(len(new_list)) if re.search('and', new_list[i]) ]
#[i for i in range(len(new_list)) if re.search('\t', new_list[i]) ]

# Eliminate bad char in text columns: title, overview, original_title
bad_cols = ['title', 'overview', 'original_title']
for c in range(len(bad_cols)):
    new_list = list(movies_data[bad_cols[c]])
    movies_data[bad_cols[c]] = [str(new_list[i]).replace("\r\t\v"," ") for i in range(len(new_list))]

#movies_data['overview'] = [str(new_list[i]).translate('', '\t') for i in range(len(new_list))]
movies_data['overview'] = [str(new_list[i]).replace("\r\t\v"," ") for i in range(len(new_list))]
movies_data['title'] = [str(list(movies_data['title'])[i]).replace("\r\t\v"," ") for i in range(len(list(movies_data['title'])))]

# Eliminate bad char ("translate")
#[old_list[i].translate(None, '\t') for i in range(len(old_list))]



# See if this throws error -> It does, suggesting that the original file all_movies was poorly formatted!
movies_data.to_csv('test_encoding.csv')
A = pd.read_csv('test_encoding.csv')

# Try skipping bad lines: this doesn't work either
B = pd.read_csv('test_encoding.csv', error_bad_lines = False)





