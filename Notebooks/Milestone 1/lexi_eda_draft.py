#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:51:50 2017

@author: AlexandraDing
"""

import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import re # Regex
import collections

# File reading packages
import pickle
import json
import csv
import datetime # For handling dates

# The "requests" library makes working with HTTP requests easier
import requests
import os
from bs4 import BeautifulSoup
from IPython.display import Image, display # Display image from URL
from IPython.core.display import HTML # Display image from URL

# TMDB API wrapper
import tmdbsimple as tmdb

# IMDB access
import imdb

# Change WD
os.chdir('/Users/AlexandraDing/Desktop/cs109b-best-group')

# Import the Datasets
top_20_genre = pd.read_csv('movie_by_genres.csv')
top_100_2016 = pd.read_csv('top_100_2016_data_with_releasedate.csv')

print(top_20_genre.head)
print(top_100_2016.head)

# Get column names
print top_20_genre.columns
print top_100_2016.columns

# Convert release_date column to datetime object
top_100_2016['release_date'] = pd.to_datetime(top_100_2016['release_date'], format = '%Y-%m-%d' )



### Fix the problem with the genres

# Get Genre correspondences and dump
url = "https://api.themoviedb.org/3/genre/movie/list?api_key=783918f9b8efd0f2ff0792d7b7de9fa2&language=en-US"

payload = "{}"
response = requests.request("GET", url, data=payload)

# reformat the result
movie_genres = response.json()

new_dict = {}
 
for item in movie_genres['genres']:
    name = item['name']
    new_dict[name] = int(item['id'])
    print(item)

# Pickle dump the directory of genres
pickle.dump(new_dict, open("genre_dict.p", 'wb'))

### Fix the problem in this dataset
n_unique_genres = len(new_dict) # Should be like 19
list_unique_genres = new_dict.values()

# Invert the dict

genre_dict_by_id = {v: k for k, v in new_dict.iteritems()}
pickle.dump(genre_dict_by_id , open("genre_dict_by_id.p", 'wb'))

## Entries are in this form: '[16, 35, 18, 10751, 10402]'
# Cnverts entries from this form: '[16, 35, 18, 10751, 10402]' -> [16, 35, 18, 10751, 10402] (list of ints)
top_100_2016['genre_ids'] = [map(int, re.sub("[\[ \] ]", "", top_100_2016['genre_ids'][i]).split(',')) for i in range(len(top_100_2016))]


# http://datascience.stackexchange.com/questions/8253/how-to-binary-encode-multi-valued-categorical-variable-from-pandas-dataframe

data_count_genres = top_100_2016['genre_ids'].apply(collections.Counter)
one_hot_encode_genres = pd.DataFrame.from_records(data_count_genres).fillna(value=int(0))
top_100_2016 = top_100_2016.join(one_hot_encode_genres)

top_100_2016.head(4)
top_100_2016.to_csv('top_100_2016_onehot_genres.csv')


#### THE ACTUAL VISUALIZATION PARTS!!!!!!

### Count genres

# N genres per entry
np.sum(one_hot_encode_genres.values, 1)

# N movies per genre
np.sum(one_hot_encode_genres.values, 0)

one_hot_encode_genres.columns

genre_names_encoded = [genre_dict_by_id[one_hot_encode_genres.columns[i]] for i in range(len(one_hot_encode_genres.columns))]
genre_counts_df = pd.DataFrame({'genre' :genre_names_encoded, 'counts': np.sum(one_hot_encode_genres.values, 0)})

genre_counts_df_sorted = genre_counts_df.sort_values('counts', ascending = False)

sns.set_context({"figure.figsize": (24, 10)})
plt.plot()
sns.barplot(x = genre_counts_df_sorted['genre'], 
            y = genre_counts_df_sorted['counts'], 
            color = "green")
plt.xlabel('Genre')
plt.ylabel('Counts')
plt.suptitle('Number of Top 100 movies of 2016 in each genre')
plt.title('Note: some movies have multiple genres')

### Get most common genre by month
# http://randyzwitch.com/creating-stacked-bar-chart-seaborn/
# First need to count genres by month
sum_genres_by_month = top_100_2016[top_100_2016.columns[-n_unique_genres+1:]].groupby(top_100_2016["release_date"].dt.month).sum()


sum_genres_month_array = np.array(sum_genres_by_month)
cumsum_genres_month_array = np.cumsum(sum_genres_month_array, 1)
#bottom = np.cumsum(ar, axis = 1)
#ind = range(1,5)
#
#plt.bar(ind, ar[0,0], color = 'b')
#for j in xrange(1, ar.shape[1]):
#    plt.bar(ind, ar[j], bottom = bottom[j-1])
#
#
## We have to do dumb things to get a stacked barplot
#f, ax1 = plt.subplots(1, figsize=(10,5))
#bar_width = 0.75


ind = np.arange(12)  # the x locations for the groups
width = 0.3  # the width of the bars

color_list = ['r', 'g', 'b', 'purple', 'yellow', 'grey', 'orange', 'r', 'g', 'b', 'r', 'g', 'b']
fig, ax = plt.subplots()
for i in reversed(range(len(ind))):
    counts_month = cumsum_genres_month_array[:,i]
    rects = ax.bar(ind, counts_month, width, color = color_list[i], alpha = 0.5)


# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_xlabel('Month')
ax.set_title('TEST')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(range(1,13))


# Plot bar plot of release date (by month)
# http://stackoverflow.com/questions/27365467/python-pandas-plot-histogram-of-dates
counts_by_release_month = top_100_2016['release_date'].groupby(top_100_2016["release_date"].dt.month).count()

plt.figure()
sns.barplot(x = counts_by_release_month.index, 
            y = counts_by_release_month.values, 
            palette="BuGn_d")
plt.xlabel('Month')
plt.ylabel('Counts')
plt.title('Number of movies in top 100 by release month, 2016')

### Image Analysis: Perhaps posters have different color mixes for different genres?





