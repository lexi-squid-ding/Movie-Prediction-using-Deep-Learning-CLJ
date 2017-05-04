#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:07:28 2017

@author: AlexandraDing
"""

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
os.chdir('/Users/AlexandraDing/Desktop/')

# My API key (Lexi)
tmdb.API_KEY = 'ee0df3ce88063f7f6cd466ff61266a55'


# Load data from 2011- 2016
#movies_data = pd.read_csv('all_movies_from_2011_2016_data.csv')
movies_data = pd.read_csv('top100pg_movies_from_2011_2016_data.csv')

movies_data.shape
print movies_data.head(4)

## See if this throws error -> It does, suggesting that the original file all_movies was poorly formatted!
#movies_data.to_csv('test_encoding.csv')
#A = pd.read_csv('test_encoding.csv')

# Try skipping bad lines: this doesn't work either
#B = pd.read_csv('test_encoding.csv', error_bad_lines = False)

###############################
### DATA PROCESSING
###############################

# Check if Some movies are missing posters or Genre IDs
print sum(pd.isnull(movies_data['poster_path']))
print np.sum(movies_data['genre_ids'] == '[]')

## Remove the entries missing Genre labels
movies_data_noempty = movies_data[movies_data['genre_ids'] != '[]']
movies_data_noempty.index = range(movies_data_noempty.shape[0])


#movies_data_noempty.to_csv('test_encoding.csv')
#A = pd.read_csv('test_encoding.csv')

#movies_data_noempty = movies_data

# Change entries to strings
movies_data_noempty['genre_ids'] = [map(int, re.sub("[\[ \] ]", "", movies_data_noempty['genre_ids'].loc[i]).split(',')) for i in range(movies_data_noempty.shape[0])]



###############################
### ONE HOT ENCODE GENRES
###############################
data_count_genres = movies_data_noempty['genre_ids'].apply(collections.Counter)
one_hot_encode_genres = pd.DataFrame.from_records(data_count_genres).fillna(value=int(0))

# Concatenate movies_data and one_hot_encode_genres
Q = pd.concat([movies_data_noempty, one_hot_encode_genres], axis = 1)


## DROP COLUMN FROM Q: Foreign (10769)

Q = Q.drop(10769, axis = 1)
print Q.shape

## Erase that entry that gives problems ("Dimmu Borgir: Forces of the Northern Night")
#Q = Q.drop([1380])

## Get entries with zero genres and erase them. 
print np.sum(np.sum(Q[Q.columns[-20:-1]], axis = 1) == 0)
Q = Q[np.sum(Q[Q.columns[-20:-1]], axis = 1) != 0]
  
# Write to CSV: top_100_2016_onehot_genres.csv

Q.to_csv('top100pg_data_2011to2016_onehot_genres.csv', index = False, encoding = 'utf-8')
# THIS LINE THROWS ERROR WHEN READING CSV
#K = pd.read_csv('top100pg_data_2011to2016_onehot_genres.csv')

### Sampling: Sample 1000 from each year randomly
# Convert release_date to datetime and get year
movies_data_all = Q
movies_data_all['release_date'] = pd.to_datetime(movies_data_all['release_date'], format = '%Y-%m-%d' )
movies_data_all['release_year'] = movies_data_all['release_date'].map(lambda x: x.year)


# Sample 1000 from each year randomly
n_sample = 1000
movies_data_sampled = pd.concat([movies_data_all[movies_data_all['release_year'] == year].sample (n = n_sample) for year in range(2011,2017)])

movies_data_sampled.to_csv('sampled_data_top100pg_2011to2016_onehot_genres.csv', index = False, encoding = 'utf-8')
#pickle.dump(movies_data_sampled, open('sampled_data_2011to2016_onehot_genres.p', 'wb' ))

M = pd.read_csv('/Users/AlexandraDing/Documents/cs109b-best-group/edited_sampled_data_top100pg_2011to2016_onehot_genres.csv')

