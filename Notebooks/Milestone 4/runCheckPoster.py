# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import CheckPoster as cp #import the self defined functions

# function to run the CheckPoster function
# start and end are the range of i on the movie-id list

def runCP(movieID, start, end, read_id_list, face_number_list, location_list):
       
    #check all the poster in the movie ID list
    #for i in range(len(movieID_AD)):
    for i in np.arange(start, end):
        #call the self-defined function
        face_number, location = cp.labelPoster(movieID[i])
    
        #store results into lists
        read_id_list.append(movieID[i])
        face_number_list.append(face_number)
        location_list.append(location)
        
        
    return read_id_list, face_number_list, location_list
    
    
