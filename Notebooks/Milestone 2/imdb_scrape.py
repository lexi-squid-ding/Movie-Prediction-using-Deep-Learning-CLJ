import numpy as np
import pandas as pd
import imdb
from imdb import IMDb
from datetime import datetime

### Function to find a movie id on IMDb using movie title and year
# input: movie title and movie year in strings
# output: matched movie id if found, otherwise return 0
def find_movie_id(movie_name, movie_year):
    # search for the movie    
    ia = IMDb()
    search_result = ia.search_movie(title)
    
    # go through the result and match to the given year
    # use the first movie matched to the year
    for movie in search_result:
        # some movies do not have year information
        # set year to 0 so guaranteed no match
        try:
            year = movie['year']
        except:
            year = 0

        if year == movie_year:
            matched_movie = movie
            break
    
    # find the movie ID
    try: 
        # find both movieID and title for double checking
        matched_id = ia.get_imdbID(matched_movie)
        matched_title = matched_movie['title']
    except:
        # id = 0 if movie not exist
        matched_id = 0
        matched_title = "NA"
        
    return(matched_id, matched_title)



### Function to find movie information using movie ID
# input: movie id from IMDb
# output: runtime, director and aspect ratio
def extract_movie_info(movie_id):
    # find the movie information
    ia = IMDb()
    movie = ia.get_movie(movie_id)
    
    # extract relevant information 
    try:
        runtime = ",".join(movie.data['runtimes'])
    except:
        runtime = "NA"
        
    try:
        director = movie.data['director'][0]['name']
    except: 
        director = "NA"
        
    try:
        aspect_ratio = movie.data['aspect ratio']
    except:
        aspect_ratio = "NA"
    
    
    return(runtime, director, aspect_ratio)




#### read the combined data
movie_data = pd.read_csv("edited_sampled_data_top100pg_2011to2016_onehot_genres.csv")




######   find movie ID for the movies   ######

#release_years = list() #data already have release year
movie_ids = list()
matched_title = list()

for i in range(movie_data.shape[0]):
    if i % 500 == 0:
        print (i)
        
    # extract movie title
    title = movie_data.iloc[i]["title"]
    
    # extract release year
    #t = movie_data.iloc[i]["release_date"]
    #year = datetime.strptime(t, '%Y-%M-%d').year
    #release_years.append(year)
    year = movie_data.iloc[i]["release_year"] 
    year = int(year)
    
    # find movie_id
    movie_id, movie_title = find_movie_id(title, year)
    movie_ids.append(movie_id)
    matched_title.append(movie_title)


# skip i = 2582 - 2591
# skip i = 2795 - 2818
#for i in np.arange(2795, 2819):
#    movie_ids.append(0)
#    matched_title.append("NA")
  
  

## mismatched due to manual correction
#movie_ids.insert(2819, 0)
#matched_title.insert(2819, "NA")


# add movie id and matched title to the dataframe
#movie_data['year'] = pd.Series(release_years, index=movie_data.index)
movie_data['imdb_id'] = pd.Series(movie_ids, index=movie_data.index)
movie_data['imdb_title'] = pd.Series(matched_title, index=movie_data.index)


# drop rows 2582-2591, 2795 - 2819 (missing movie info or format messed up)
movie_data = movie_data.drop(np.arange(2582, 2592), axis=0)   
movie_data = movie_data.drop(np.arange(2795, 2820), axis=0)   




######   find movie information on IMDb   #######

# read the movie id due to deleted rows
movie_ids = movie_data["imdb_id"]

movie_runtimes = list()
movie_directors = list()
movie_ratio = list()

i = 0  # to track the progress

for movie_id in movie_ids:
    if i % 100 == 0:
        print (i)
    i=i+1
        
    # if not empty
    if movie_id != 0:
        runtime, director, aspect_ratio = extract_movie_info(movie_id)
    else: # NA if empty
        runtime = director = aspect_ratio = "NA"
    
    # add to the lists
    movie_runtimes.append(runtime)
    movie_directors.append(director)
    movie_ratio.append(aspect_ratio)
  
  
# add runtime, director and aspect ratio to the dataframe
movie_data['runtime'] = pd.Series(movie_runtimes, index=movie_data.index)
movie_data['director'] = pd.Series(movie_directors, index=movie_data.index)
movie_data['aspect_ratio'] = pd.Series(movie_ratio, index=movie_data.index)
 
       
       
# save the dataframe into csv
movie_data.to_csv("edited_sampled_data_top100pg_2011to2016_onehot_imdb.csv", 
                  index=False)
movie_data.to_pickle("add_imdb")