import numpy as np
import pandas as pd
import requests
import urllib
import time



# read the movie data for the poster path
data = pd.read_csv("poster_urls_Top1000_each_2005-2016.csv")

# poster path base url (poster width = 92)
base_url = "http://image.tmdb.org/t/p/w92/"

# download posters for each movie 
for i in range(data.shape[0]):
    
    if i % 500 == 0:
        print i

    # add the poster path to url
    try:
        url = base_url + data["poster_path"][i]
    except:
        url = base_url
     
    # name the poster by movie id
    filename = "posters/" + str(data["id"][i]) + ".jpg"
    
    # download the image
    image = urllib.URLopener()

    try:
        image.retrieve(url, filename)
    except:
        print ("No poster")
        
    time.sleep(2)
