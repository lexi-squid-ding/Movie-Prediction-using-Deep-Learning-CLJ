#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:50:10 2017

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

# TMDB API wrapper
import tmdbsimple as tmdb

os.chdir('/Users/AlexandraDing/Documents/cs109b-best-group')

# Load CSV with imdb data added
movie_df = pd.read_csv('add_imdb_utf8.csv')

# look at runtime
#runtime_nomissing = movie_df['runtime'][movie_df['runtime'].isnull()]
runtime_nomissing = movie_df['runtime'].dropna()

# Source: http://stackoverflow.com/questions/1059559/split-strings-with-multiple-delimiters
runtime_splits = [re.findall(r"[\w']+", runtime_nomissing.iloc[i]) for i in range(len(runtime_nomissing))]


runtime_value = np.zeros((len(runtime_nomissing)))

for j in range(len(runtime_splits)):
    list_num = [int(s) for s in runtime_splits[j] if s.isdigit()]
    runtime_value[j] =list_num[0]

# Reassign runtime_value to the correct column
movie_df['runtime'].iloc[runtime_nomissing.index] = runtime_value

### Save file with fixed runtime
movie_df.to_csv('add_imdb_utf8_fixruntime.csv', index = False)

##### Now work on the problem with the aspect ratios:

# Explore Unique Values
A = movie_df['aspect_ratio'].unique()

for i in range(len(A)):
    print A[i]

# Decided to encode as CATEGORICAL VARIABLE



