# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 09:04:20 2017

@author: Nguyen
"""

#Data Frames
import pandas as pd

#Linear Algebra
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Image Manipulation
from scipy import signal

def show_images(arrays, kernel=None):
    fig = plt.figure(1, figsize=(20,20))
    for i in range(9):
        ax = fig.add_subplot(3,3,i+1)
        arr = signal.convolve2d(np.reshape(np.array(arrays.iloc[i,0]), (75,75)), kernel, mode='valid')
        ax.imshow(arr, cmap = 'inferno')

if __name__ == '__main__':
    train = pd.read_json('../Data/train.json')
    train['inc_angle'] = pd.to_numeric(train['inc_angle'], errors = 'coerce')
    print(train.head())
    #Some practice from https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python
    icebergs = train[train.is_iceberg==1].sample(n=9, random_state = 456)
    not_icebergs = train[train.is_iceberg==0].sample(n=9, random_state = 123)
    
    #Image kernels (i.e. edge detectors like Sobel)
    xder = np.array([[-1,0,1], [-2,0,2],[-1,0,1]])
    yder = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    smooth = np.array([[1,1,1],[1,10,1],[1,1,1]])
    xder2 = np.array([[-1,2,-1],[-3,6,-3],[-1,2,-1]])
    yder2 = np.array([[-1,-3,-1],[2,6,2],[-1,-3,-1]])
    show_images(icebergs,smooth) 