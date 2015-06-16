# -*- coding: utf-8 -*-
"""
A short script for plotting data
"""

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec #fancy subplot layout

def maxIntProj(data, **kwargs):
    """
    A wrapper function for the two sub methods: 
    * _maxIntProj2D
    * _maxIntProj3D
    Based on the size of data
    
    Parameters
    ----------------
    data =  the data to be plotted
    **kwargs = the kwargs to be passed along
    
    Returns
    ----------------
    fig = figure handle
    ax = axes handles in a list
    """
    
    numdims = data.ndim
    print(numdims)
    if(numdims == 2):
        print('2d data')
        return _maxIntProj2D(data, **kwargs)
    elif(numdims == 3):
        print('3d data')
        return _maxIntProj3D(data, **kwargs)
    else:
        raise TypeError('Data was not two dimensional or three dimensional')


def _maxIntProj2D(data, takelog=False):
    #set up the grid for the subplots
    gs = gridspec.GridSpec(2,2,width_ratios=[1, myShape[0]/myShape[2]],
                           height_ratios=[1, myShape[0]/myShape[1]])
    
    #set up my canvas
    fig = plt.figure(figsize=(5,5)) #necessary to make the overall figure shape square,
                            #without this the boxes aren't sized properly
    # Max X and Y
    maxY = np.amax(data,axis=0)    
    maxX = np.amax(data,axis=1)
        
    #set up each projection
    ax_XY = plt.subplot(gs[2])
    ax_XY.matshow(data)
    ax_XY.set_title('XY')
    ax_XY.axis('tight')
    
    ax_Y = plt.subplot(gs[3], sharey=ax_XY)
    ax_Y.plot(maxX,np.arange(data.shape[0]))
    ax_Y.set_title('Y')
    ax_Y.axis('tight')
    
    ax_X = plt.subplot(gs[0], sharex=ax_XY)
    ax_X.plot(maxY)
    ax_X.set_title('X')
    ax_X.axis('tight')
    
    fig.tight_layout()
    
    
def _maxIntProj3D(data, takelog=False):
    print('Plotting 3D data')
    maxZ = np.amax(data, axis=0)
    maxY = np.amax(data, axis=1)
    maxX = np.amax(data, axis=2)
    
    #grab the data shape
    myShape = data.shape #get the shape of my PSF
    #set up the grid for the subplots
    gs = gridspec.GridSpec(2,2,width_ratios=[1, myShape[0]/myShape[2]],
                           height_ratios=[1, myShape[0]/myShape[1]])
    #set up my canvas
    fig = plt.figure(figsize=(5,5)) #necessary to make the overall figure shape
    #square, without this the boxes aren't sized properly
    
    #set up each projection
    ax_XY = plt.subplot(gs[0])
    ax_XY.matshow(maxZ)
    ax_XY.set_title('XY')
    ax_XY.axis('tight')
    
    ax_YZ = plt.subplot(gs[1], sharey=ax_XY)
    ax_YZ.matshow(maxX.T)
    ax_YZ.set_title('YZ')
    ax_YZ.axis('tight')
    
    ax_XZ = plt.subplot(gs[2], sharex=ax_XY)
    ax_XZ.matshow(maxY)
    ax_XZ.set_title('XZ')
    ax_XZ.axis('tight')
    
    fig.tight_layout()
    fig.show()
    
#scritping ability
if __name__=='main':
    #Need to take the first system argument as the filename for a TIF file
    import sys
    print('Running',sys.argv[0],'on',sys.argv[1])
    #test if filename has tiff in it
    filename = sys.argv[1] #assume that first argument is file name
    if '.tif' in filename or '.tiff' in filename:
        #Import skimage so we have access to tiff loading
        from skimage.external import tifffile as tif
        data = tif.imread(filename)
        maxIntProj(data)
    else:
        print('nothing to see here')
        pass