# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import cv2  #package for image display
from matplotlib import pyplot as plt
from Tkinter import *  #package for interactive GUI widget
import tkSimpleDialog  #interactive dialogue box

#create a multiple choice widget to collect answers for title location
def getTitleLoc():
    #set GUI
    root = Tk()
    #set variable as string
    v = StringVar()
    
    #set title label of the GUI 
    Label(root, text="What is the location of the title?", justify = LEFT, padx = 20).pack()
    #create four multiple choices
    Radiobutton(root, text="Top", padx = 20, variable=v, value="Top").pack(anchor=W)
    Radiobutton(root, text="Middle", padx = 20, variable=v, value="Middle").pack(anchor=W)            
    Radiobutton(root, text="Bottom", padx = 20, variable=v, value="Bottom").pack(anchor=W)
    Radiobutton(root, text="Other",padx = 20, variable=v, value="Other").pack(anchor=W)
            
    #exit the pop-up window
    root.mainloop()
   
    #read selection
    loc = v.get()
   
    return loc

#create a function to collect image information through the following steps:
#1.pop up impage according to input movie ID
#2.collect number of faces via dialogue box
#3.collect title location via a multiple choice widget

def labelPoster(movieID):

    flag = 0
    #for movies with poster
    try:
        #construct movie poster file path
        path = 'posters_2005_2010/' + str(movieID) + '.jpg'
        #read image
        img = cv2.imread(path)
        #hide tick values on X and Y axis
        plt.axis("off")
        plt.imshow(img)
        #show image in a separate window
        plt.show()
    except:
        flag = 1
        
        
    if flag == 0:
        #Set up our GUIs
        root = Tk()
        #collect face number 
        face_number = tkSimpleDialog.askinteger("Number of Faces", "How many faces are in the poster?")
        #collect title location
        location = tkSimpleDialog.askstring("Title position",
                                            "What is position of the title?(Top(t), Middle(m), Bottom(b), Others(o))")
        #location = getTitleLoc() 
        
        #exit the pop-up window
        root.mainloop()
        #plt.close()
        
    #for movies without poster
    else:
        face_number = "missingval"
        location = "missingval"
            
    return face_number, location