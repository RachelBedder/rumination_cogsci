import numpy as np
import matplotlib as mpl
import os
import pickle

def index_finder(x,y): 
    arr = np.abs(x-y) 
    mindex = np.argmin(arr)
    return mindex

def get_subfolders(parent_folder):
    subfolders = []
    for foldername in os.listdir(parent_folder):
        folderpath = os.path.join(parent_folder, foldername)
        if os.path.isdir(folderpath):
            subfolders.append(foldername)
    return subfolders

def pickle_shelf(parent_name, file_name):
    file_path = os.path.join(parent_name, 'objects', file_name)
    with open(file_path, 'rb') as f:
        # deserialize the list of objects from the file
        obj_list = pickle.load(f)
    return obj_list