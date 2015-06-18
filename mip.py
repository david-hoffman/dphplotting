# -*- coding: utf-8 -*-
"""
TO DO
------------------
- Autoadjust whitespace
- Add option to define pixel size `(x,y)` and `(z)` independently
-
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec #fancy subplot layout

def mip(data, **kwargs):
    """
    A wrapper function for the two sub methods:

    * _mip2D
    * _mip3D

    Chooses the correct one based on the size of data

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
        return _mip2D(data, **kwargs)
    elif(numdims == 3):
        return _mip3D(data, **kwargs)
    else: #raise a type error if the data is not 2 or 3D
        raise TypeError('Data was not two dimensional or three dimensional')


def _mip2D(data, takelog=False):
    """
    A subfunction that makes a nice plot of 2D data with max projections along either side

    Parameters
    ----------------
    data = the data passed from the mip wrapper function
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


def _mip3D(data, takelog=False):
    """
    A subfunction that makes a nice plot of 3D data with max projections along either side

    Documentation very similar for `_mip2D`

    Parameters
    ----------------
    data = the data passed from the mip wrapper function
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
