#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import re
from datetime import datetime
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import sklearn.preprocessing as Preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA




# read the data
data = pd.read_csv("add_imdb_utf8_fixruntime_cleaned.csv")
data.shape

# eliminate movies without genre_ids
data = data[pd.notnull(data["genre_ids"])]
data.shape



######    extract and clean predictors     ######

##### quantitative predictors

quant_cols = ["popularity", "vote_count" , "runtime", "vote_average"]


### check proportion of missing data
for col in quant_cols:
    print (col)
    print (np.mean(pd.isnull(data[col])))


### read the predictors and convert string into float
# convert df to array
x_quant = data[quant_cols].values

# convert each column to float
for i in range(x_quant.shape[1]):
    x_quant[:, i] = map(float, x_quant[:, i])


### mean imputation on the missing values

for i in range(x_quant.shape[1]):
    col_mean = np.nanmean(x_quant[:, i])
    x_quant[np.isnan(x_quant[:, i]), i] = col_mean



##### categorical predictors

### add release month to the data
release_month = list()

for i in range(data.shape[0]):
    # extract release year
    t = data.iloc[i]["release_date"]
    month = datetime.strptime(t, '%m/%d/%Y').month
    release_month.append(month)
    
# add to the data
data["release_month"] = pd.Series(release_month, index=data.index)


### one-hot-encode categorial variables
cat_cols = ["director", "aspect_ratio", "release_year", "release_month"]


# encode directors first
x_directors = pd.get_dummies(data["director"])
# how many directors in total?
x_directors.shape  
# number of movies for each director
n_movie_director = x_directors.sum(axis = 1)
# check the histogram
plt.hist(n_movie_director)
# cannot use since no director have more than 1 movie


# encode the other columns
# convert year and month to strings
data["release_year"] = map(str, data["release_year"].values)
data["release_month"] = map(str, data["release_month"].values)

cat_cols = cat_cols[1:len(cat_cols)]
x_cat = pd.get_dummies(data[cat_cols], drop_first=True)


##### combine quant and cat predictors

# predictor array with the metadata
x_array = np.concatenate((x_quant, x_cat.values), axis = 1)
x_array.shape

# name of the predictors
x_names = np.concatenate((quant_cols, x_cat.columns), axis = 0)




#######       add the text related predictors      #######

# create another copy of the data in order to merge the code
movie_data = data

# apply bag-of-words to movie titles
# eliminate common stop words
vectorizer = CountVectorizer(stop_words='english')

# apply to movie titles
corpus_title = movie_data['title'].values
# apply to movie overview
corpus_overview = movie_data['overview'].values


### obtain the index that caused the encoding error
bad_title_index = []
for i in range(len(corpus_title)):
    try:
        corpus_title[i].encode('utf-8')
    except:
        bad_title_index.append(i)
        
        
### obtain the index that caused the encoding error
bad_ov_index = []
for i in range(len(corpus_overview)):
    try:
        corpus_overview[i].encode('utf-8')
    except:
        bad_ov_index.append(i)
        

### find the index of movie titles with nan
null_title_index = []

for i in range(len(corpus_title)):
    if str(corpus_title[i]) == "nan":
        null_title_index.append(i)
        
print null_title_index
#check if the index is correct
print corpus_title[null_title_index]

#encode nan as missingval
#use 'missingval' but not 'missing' because 'missing' can really appear in movie titles
corpus_title[null_title_index] = 'missingval'
#check if encoding is correct
print corpus_title[null_title_index]


### find the index of movie titles with nan
null_ov_index = []

for i in range(len(corpus_overview)):
    if str(corpus_overview[i]) == "nan":
        null_ov_index.append(i)
        
print null_ov_index
#check if the index is correct
print corpus_overview[null_ov_index]

#encode nan as missingval
#use 'missingval' but not 'missing' because 'missing' can really appear in movie titles
corpus_overview[null_ov_index] = 'missingval'
#check if encoding is correct
print corpus_overview[null_ov_index]


### remove bad words
for i in bad_title_index:
    bad_title = corpus_title[i].split()
    for s in bad_title:
        try:
            m = s.encode('utf-8')
        except:
            bad_title.remove(s)
    
    #re-join the words with the problematic words being removed
    corpus_title[i] = " ".join(bad_title)          
    print corpus_title[i]  
    

bad_title_unfixed_index =[]
for i in range(len(corpus_title)):
    try:
        corpus_title[i].encode('utf-8')
    except:
        print corpus_title[i]
        bad_title_unfixed_index.append(i)
        
        
### count words in title
title_counts = vectorizer.fit_transform(corpus_title)
# convert to array
title_counts = title_counts.toarray()

# word list for title
title_words = vectorizer.get_feature_names()
title_words = np.asarray(title_words)        

# convert to df
title_counts_df = pd.DataFrame(title_counts)
title_counts_df.columns = title_words
title_counts_df.head()


### append the count for each word at the end
word_count = title_counts_df.sum(axis=0)
title_counts_df = title_counts_df.append(word_count, ignore_index=True)
#change the index name to be more informative
title_counts_df=title_counts_df.rename(index = {title_counts_df.shape[0]:'sum'})


### sort dataframe by the value of the last row
new_columns = title_counts_df.columns[title_counts_df.ix[title_counts_df.last_valid_index()].argsort()]
title_counts_df = title_counts_df[new_columns]
#reverse the columns order
title_counts_df = title_counts_df[title_counts_df.columns[::-1]]



### clean the overview data 10 times
for j in range(10):
    bad_ov_unfixed_index =[]
    for i in range(len(corpus_overview)):
        try:
            corpus_overview[i].encode('utf-8')
        except:
            bad_ov_unfixed_index.append(i)

    for k in bad_ov_unfixed_index:
        bad_overview = corpus_overview[k].split()
        for s in bad_overview:
            try:
                m = s.encode('utf-8')
            except:
                bad_overview.remove(s)
        #re-join the words with the problematic words being removed
        corpus_overview[k] = " ".join(bad_overview)  



### count words for movie overview 
overview_counts = vectorizer.fit_transform(corpus_overview)
# convert to array
overview_counts = overview_counts.toarray()

# word list for title
overview_words = vectorizer.get_feature_names()
overview_words = np.asarray(overview_words)

overview_counts_df = pd.DataFrame(overview_counts)
overview_counts_df.columns = overview_words
overview_counts_df.head()

# convert to df
overview_word_count = overview_counts_df.sum(axis=0)
#append the count for each word at the end
overview_counts_df = overview_counts_df.append(overview_word_count, ignore_index=True)
overview_counts_df.shape

#change the index name to be more informative
overview_counts_df=overview_counts_df.rename(index = {overview_counts_df.shape[0]:'sum'})

#sort dataframe by the value of the last row
ov_new_columns = overview_counts_df.columns[overview_counts_df.ix[overview_counts_df.last_valid_index()].argsort()]
overview_counts_df = overview_counts_df[ov_new_columns]
#reverse the columns order
overview_counts_df = overview_counts_df[overview_counts_df.columns[::-1]]
overview_counts_df.tail()



##### apply PCA on the text counts

pca = PCA()

### on movie title
title_pca = pca.fit_transform(title_counts_df.iloc[:-1])
# find the #PC explain 90% var
total_var = np.cumsum(pca.explained_variance_ratio_)
n_pc_title = np.where((total_var > 0.9) == True)[0][0]

### on movie overview
overview_pca = pca.fit_transform(overview_counts_df[:-1])
total_var = np.cumsum(pca.explained_variance_ratio_)
# find the #PC explain 90% var
n_pc_overview = np.where((total_var > 0.9) == True)[0][0]

# does not reduce much of the dimensionality
    
    
    
### combine test data 
    
# save the first 66 columns of the overview (>200 times)
selected_overview_counts = overview_counts_df.iloc[:-1, :66].values

# save the first 17 columns of the title (>30 times)
selected_title_counts = title_counts_df.iloc[:-1, :17].values

#combine with x_array  
x_array = np.concatenate((x_array, selected_title_counts, selected_overview_counts), axis = 1)




######   extract and clean the response variable   ######

# convert genre ids to list
y = list()
for i in range(data.shape[0]):
    genre = map(int, re.sub("[\[ \] ]", "", data['genre_ids'][i]).split(','))
    y.append(genre)

# binarize response variable (returns array)
y_binary = MultiLabelBinarizer().fit_transform(y)




### PREPROCESSING: Scale continuous features 
standardize = Preprocessing.StandardScaler()
x_std = standardize.fit_transform(x_array)

features_targets_dump = list([x_array, x_std, y_binary])
pickle.dump(features_targets_dump, open('continuous_features_targets.p', 'wb'))




#######       Model Fitting        ######: Done in a separate file
#
## initiate the model with linear SVC
#OVR_svc = OneVsRestClassifier(LinearSVC(random_state=0))
#
## fit the model
#OVR_svc.fit(x_array, y_binary)
#
## predict on something
#Y_pred = OVR_svc.predict(X_meta_array)


