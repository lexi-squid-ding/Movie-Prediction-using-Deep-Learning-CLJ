import numpy as np
import pandas as pd
import runCheckPoster as rc
import pickle

#Lexi's movie 0:2000

#Cynthia's movie: 2000:4000

#Jingyi's movie: 4000:6000



#load data
data = pd.read_csv("posters_2005_2010/movie_posters_list.csv")
#store movie ID
movieID = data["id"][4000:].values

#### run this part only when at the beginning!
face_number_list = list()
location_list = list()
read_id_list = list()


#### manually change this number
start = 0  #lexi start at 0, Cynthia at 2000, and Jingyi at 4000


### repeat the following
# first check the length of the results and the start number
print "There are {} movies scored. The current number is {}".format(len(read_id_list), start)

# read the next 50 movies
end = start + 50

read_id_list, face_number_list, location_list = rc.runCP(movieID, start, end,
                                                    read_id_list,
                                                    face_number_list, 
                                                    location_list)

# save the result so far
pickle.dump([read_id_list, face_number_list, location_list], 
            open('label_results.p', 'wb'))

start = end


### read previous saved results
result = pd.read_pickle("label_results.p")
[read_id_list, face_number_list, location_list] = result

start = 1100


### save the result into csv
poster_label = pd.DataFrame({
                            'id': read_id_list,
                            'face_count': face_number_list,
                            'title_location': location_list
})

poster_label.to_csv("posters_label_JY.csv", index = False)




#---------------------------------------
### combine final scored results

# read the results
result_JY = pd.read_csv("posters_label_JY.csv")
result_CH = pd.read_csv("CH_movie_posters_list.csv")
result_LD = pd.read_csv("posters_2005_2010/movie_posters_list_Lexi.csv")

# extract relevant columns
result_CH = result_CH.iloc[:, np.arange(1, 4)]
result_CH.columns = result_JY.columns

result_LD = result_LD[result_JY.columns]


# combine the results
result = pd.concat([result_LD, result_CH, result_JY], axis = 0)

# save the result
result.to_csv("final_poster_label_2005_2010.csv", index = None)