# -*- coding: utf-8 -*-
"""
TO DO
------------------
- Autoadjust whitespace
- Add option to define pixel size `(x,y)` and `(z)` independently
-
"""

import numpy as np
#import scipy as sp
import matplotlib
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
    ax = axes handles in a flat ndarray
    """

    #pull the number of dimensions of the data
    #so that we can call the correct sub function
    numdims = data.ndim

    if(numdims == 2):
        return _maxIntProj2D(data, **kwargs)
    elif(numdims == 3):
        return _maxIntProj3D(data, **kwargs)
    else: #raise a type error if the data is not 2 or 3D
        raise TypeError('Data was not two dimensional or three dimensional')


def _maxIntProj2D(data, takelog=False):
    """
    A subfunction that makes a nice plot of 2D data with max projections along either side

    Parameters
    ----------------
    data = the data passed from the maxIntProj wrapper function
    takelog = not implemented yet

    Returns
    ----------------
    fig = figure handle
    ax = axes handles in a flat ndarray
    """

    #figure out the shape of the data
    myShape = data.shape

    #use that data to correctly scale the grids that we'll plot on
    gs = gridspec.GridSpec(2,2,width_ratios=[2,1],
                           height_ratios=[1, 2])

    #set up my canvas
    fig = plt.figure(figsize=(5*myShape[1]/myShape[0],5)) #necessary to make the overall
                            #figure shape square, without this the boxes aren't sized
                            #properly

    # Max X and Y
    maxY = np.amax(data,axis=0)
    maxX = np.amax(data,axis=1)

    #set up each projection
    #Z
    ax_XY = plt.subplot(gs[2])
    ax_XY.matshow(data)
    ax_XY.set_title('XY')
    ax_XY.axis('tight')

    #Y
    ax_Y = plt.subplot(gs[3], sharey=ax_XY)
    ax_Y.plot(maxX,np.arange(data.shape[0]))
    ax_Y.set_title('Y')
    ax_Y.axis('tight')

    #X
    ax_X = plt.subplot(gs[0], sharex=ax_XY)
    ax_X.plot(maxY)
    ax_X.set_title('X')
    ax_X.axis('tight')

    #The MacOSX backend cannot be used interactively with `tight_layout()`
    if(matplotlib.get_backend() != 'MacOSX'):
        fig.tight_layout()

    return fig, np.array([ax_XY, ax_Y, ax_X])


def _maxIntProj3D(data, takelog=False):
    """
    A subfunction that makes a nice plot of 3D data with max projections along either side

    Documentation very similar for `_maxIntProj2D`

    Parameters
    ----------------
    data = the data passed from the maxIntProj wrapper function
    takelog = not implemented yet

    Returns
    ----------------
    fig = figure handle
    ax = axes handles in a flat ndarray
    """
    #print('Plotting 3D data')
    maxZ = np.amax(data, axis=0)
    maxY = np.amax(data, axis=1)
    maxX = np.amax(data, axis=2)

    #grab the data shape
    myShape = data.shape

    #set up the grid for the subplots
    gs = gridspec.GridSpec(2,2,width_ratios=[1, myShape[0]/myShape[2]],
                           height_ratios=[1, myShape[0]/myShape[1]])
    #set up my canvas
    fig = plt.figure(figsize=(5*myShape[2]/myShape[1],5)) #necessary to make the overall
                                                        #figure shape square, without this
                                                        #the boxes aren't sized properly

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

    if(matplotlib.get_backend() != 'MacOSX'):
        fig.tight_layout()

    return fig, np.array([ax_XY, ax_YZ, ax_XZ])

#Here we add scripting ability so that this module may be called as a standalone script
#from the command line
if __name__=='__main__': #check to see if we're being run from the command line
    import sys #we need access to the system, especially argv

    #if we want to update this to take more arguments we'll need to use one of the
    #argument parsing packages

    #a little output so that the user knows whats going on
    print('Running',sys.argv[0],'on',sys.argv[1])

    #Need to take the first system argument as the filename for a TIF file

    #test if filename has tiff in it
    filename = sys.argv[1] #assume that first argument is file name
    if '.tif' in filename or '.tiff' in filename:
        #Import skimage so we have access to tiff loading
        from skimage.external import tifffile as tif
        data = tif.imread(filename)
        #plot the data
        fig, ax = maxIntProj(data)
        #readjust the white space (maybe move this into main code later)
        fig.subplots_adjust(top=0.85,hspace=0.3,wspace=0.3)
        #add an overall figure title that's the file name
        fig.suptitle(filename, fontsize=16)
        plt.show()
    else:
        print('You didn\'t give me a TIFF')
