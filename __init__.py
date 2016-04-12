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


def display_grid(data, showcontour=False, contourcolor='w', filter_size=None,
                 figsize=3, auto=False, nrows=None, **kwargs):
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
    if nrows is None:
        nrows = int(np.sqrt(numitems))
    ncols = int(np.ceil(numitems/nrows))

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(figsize*ncols, figsize*nrows))
    for (k, v), ax in zip(sorted(data.items()), axs.ravel()):
        if v.ndim == 1:
            ax.plot(v, **kwargs)
        else:
            if auto:
                # pop vmin and vmax from **kwargs
                kwargs.pop('vmin', None)
                kwargs.pop('vmax', None)
                # calculate vmin, vmax
                vmin, vmax = auto_adjust(v)
                ax.imshow(v, vmin=vmin, vmax=vmax, **kwargs)
            else:
                ax.imshow(v, **kwargs)
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


def recolor(ax, cmap, new_alpha=None):
    '''
    Recolor the lines in ax with the cmap
    '''
    # figure out how many lines are in ax
    num_lines = len(ax.lines)
    # set the new alpha mapping, if wanted
    if new_alpha is not None:
        if 'best' in new_alpha:
            r = 1/num_lines
            try:
                expon = new_alpha['best']
            except TypeError:
                expon = 2
            new_alpha = 1 - ((1 - np.sqrt(r))/(1 + np.sqrt(r)))**expon
    # cycle through colors and recolor lines
    for i, line in enumerate(ax.lines):
        # generate new color
        new_color = list(cmap(i/num_lines))
        # replace alpha is wanted
        if new_alpha is not None:
            new_color[-1] = new_alpha
        # set the color
        line.set_c(new_color)


def auto_adjust(img, nbins=256):
    '''
    Python translation of ImageJ autoadjust function

    Parameters
    ----------
    img : ndarray

    Returns
    -------
    (vmin, vmax) : tuple of numbers
    '''
    # calc statistics
    pixel_count = int(np.array((img.shape)).prod())
    # get image statistics
    # ImageStatistics stats = imp.getStatistics()
    # initialize limit
    limit = pixel_count/10
    # histogram
    my_hist, bins = np.histogram(img.ravel(), bins=nbins)
    # convert bin edges to bin centers
    bins = np.diff(bins) + bins[:-1]
    # set up the threshold
    # Below is what ImageJ purportedly does.
    # auto_threshold = threshold_isodata(img, nbins=bins)
    # if auto_threshold < 10:
    #     auto_threshold = 5000
    # else:
    #     auto_threshold /= 2
    # this version, below, seems to converge as nbins increases.
    threshold = pixel_count/(nbins*16)
    # find the minimum by iterating through the histogram
    # which has 256 bins
    valid_bins = bins[np.logical_and(my_hist < limit, my_hist > threshold)]
    # check if the found limits are valid.
    try:
        vmin = valid_bins[0]
        vmax = valid_bins[-1]
    except IndexError:
        vmin = 0
        vmax = 0

    if vmax <= vmin:
        vmin = img.min()
        vmax = img.max()

    return (vmin, vmax)
