#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:38:11 2017

@author: AlexandraDing
"""

# Create master excel file

import os
import pandas as pd

os.chdir('/Users/AlexandraDing/Documents/cs109b-best-group/posters_2005_2010')

list_poster_files = os.listdir('/Users/AlexandraDing/Documents/cs109b-best-group/posters_2005_2010')

# 5963 posters: strip string of 
list_poster_ids = [int(item.replace('.jpg', '')) for item in list_poster_files]
print list_poster_ids


# Save as CSV using Pandas
list_unordered = pd.Series(data = list_poster_ids)
list_ordered = list_unordered.sort_values()
list_ordered.index = range(len(list_unordered))
print list_ordered
list_ordered.to_csv('movie_posters_list.csv')
