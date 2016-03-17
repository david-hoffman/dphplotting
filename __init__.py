# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:31:00 2015

@author: david
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# fancy subplot layout
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
from dphutils import auto_adjust


def display_grid(data, showcontour=False, contourcolor='w', filter_size=None,
                 figsize=3, auto=False, **kwargs):
    '''
    Display a dictionary of images in a nice grid

    Parameters
    ----------
    data : dict
        a dictionary of images
    showcontour: bool (default, False)
        Whether to show contours or not
    '''
    if not isinstance(data, dict):
        raise TypeError('Data is not a dictionary!')

    numitems = len(data)
    nrows = int(np.sqrt(numitems))
    ncols = int(np.ceil(numitems/nrows))

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(figsize*ncols, figsize*nrows))
    for (k, v), ax in zip(sorted(data.items()), axs.ravel()):
        if v.ndim == 1:
            ax.plot(v, **kwargs)
        elif v.ndim == 2:
            if auto:
                # pop vmin and vmax from **kwargs
                kwargs.pop('vmin', None)
                kwargs.pop('vmax', None)
                # calculate vmin, vmax
                vmin, vmax = auto_adjust(v)
                ax.matshow(v, vmin=vmin, vmax=vmax, **kwargs)
            else:
                ax.matshow(v, **kwargs)
            if showcontour:
                if filter_size is None:
                    vv = v
                else:
                    vv = gaussian_filter(v, filter_size)

                ax.contour(vv, colors=contourcolor)
        ax.set_title(k)
        ax.axis('off')

    for ax in axs.ravel():
        if not (len(ax.images) or len(ax.lines)):
            fig.delaxes(ax)

    return fig, axs


def slice_plot(data, center=None, allaxes=False, **kwargs):
    '''
    Parameters
    ----------
    data : ndarray
        the data passed from the mip wrapper function
    allaxes : bool (default, False)
        A boolean that determines if all the axes should be returned of if the
        middle one should be deleted

    Returns
    -------
    fig : ndarray
        figure handle
    ax : ndarray
        axes handles in a flat ndarray
    '''

    if center is None:
        center = (np.array(data.shape)/2).round().astype(int)
    else:
        center = np.array(center, dtype=int)

    maxZ = data[center[0]]
    maxY = data[:, center[1]]
    maxX = data[:, :, center[2]]

    # grab the data shape
    myShape = data.shape

    # set up the grid for the subplots
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, myShape[0]/myShape[2]],
                           height_ratios=[1, myShape[0]/myShape[1]])
    # set up my canvas
    # necessary to make the overall
    # figure shape square, without this
    # the boxes aren't sized properly
    fig = plt.figure(figsize=(5*myShape[2]/myShape[1], 5))

    # set up each projection
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

    if allaxes:
        return fig, np.array([ax_XY, ax_YZ, ax_XZ, plt.subplot(gs[3])])
    else:
        return fig, np.array([ax_XY, ax_YZ, ax_XZ])


def recolor(ax, cmap):
    '''
    Recolor the lines in ax with the cmap
    '''
    # figure out how many lines are in ax
    lines = [item for item in ax.get_children()
             if isinstance(item, matplotlib.lines.Line2D)]
    # recolor
    for line, color in zip(lines, cmap(np.linspace(0, 1, len(lines)))):
        line.set_color(color)
