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
from dphutils import fft_gaussian_filter
try:
    from pyfftw.interfaces.numpy_fft import rfftn, rfftfreq
    import pyfftw
    # Turn on the cache for optimum performance
    pyfftw.interfaces.cache.enable()
except ImportError:
    from numpy.fft import rfftn, rfftfreq

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
    ncols = int(np.ceil(numitems / nrows))

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(figsize * ncols, figsize * nrows))
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
                    vv = fft_gaussian_filter(v, filter_size)

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
        center = (np.array(data.shape) / 2).round().astype(int)
    else:
        center = np.array(center, dtype=int)

    maxZ = data[center[0]]
    maxY = data[:, center[1]]
    maxX = data[:, :, center[2]]

    # grab the data shape
    myShape = data.shape

    # set up the grid for the subplots
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, myShape[0] / myShape[2]],
                           height_ratios=[1, myShape[0] / myShape[1]])
    # set up my canvas
    # necessary to make the overall
    # figure shape square, without this
    # the boxes aren't sized properly
    fig = plt.figure(figsize=(5 * myShape[2] / myShape[1], 5))

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


def recolor(cmap, ax=None, new_alpha=None, to_change='lines'):
    '''
    Recolor the lines in ax with the cmap
    '''
    if ax is None:
        ax = plt.gca()
    # figure out how many lines are in ax
    objs = getattr(ax, to_change)
    num_objs = len(objs)
    # set the new alpha mapping, if wanted
    if new_alpha is not None:
        if 'best' == new_alpha:
            r = 1 / num_objs
            try:
                expon = new_alpha['best']
            except TypeError:
                expon = 2
            new_alpha = 1 - ((1 - np.sqrt(r)) / (1 + np.sqrt(r)))**expon
    # cycle through colors and recolor lines
    for i, obj in enumerate(objs):
        # generate new color
        new_color = list(cmap(i / num_objs))
        # replace alpha is wanted
        if new_alpha is not None:
            new_color[-1] = new_alpha
        # set the color
        obj.set_color(new_color)


def drift_plot(fit, title=None, dt=1.0, lf=-np.inf, hf=np.inf, log=False,
               cmap='magma', xc='b', yc='r'):
    '''
    Plotting utility to show drift curves nicely

    Parameters
    ----------
    fit : pandas DataFrame
        Assumes that it has attributes x0 and y0
    title : str (optional)
        Title of plot
    dt : float (optional)
        Sampling rate of data in seconds
    lf : float (optional)
        Low frequency cutoff for fourier plot
    hf : float (optional)
        High frequency cutoff for fourier plot
    log : bool (optional)
        Take logarithm of FFT data before displaying
    cmap : string or matplotlib.colors.cmap instance
        Color map for scatter plot
    xc : string or `color` instance
        Color for x data
    yc : string or `color` instance
        Color for y data

    Returns
    -------
    fig : figure object
        The figure
    axs : tuple of axes objects
        In the following order, Real axis, FFT axis, Scatter axis
    '''
    # set up plot
    fig = plt.figure()
    fig.set_size_inches(8, 4)
    axreal = plt.subplot(221)
    axfft = plt.subplot(223)
    axscatter = plt.subplot(122)
    # label it
    if title is not None:
        fig.suptitle(title, y=1.02, fontweight='bold')
    # detrend mean
    ybar = fit.y0 - fit.y0.mean()
    xbar = fit.x0 - fit.x0.mean()
    # Plot Real space
    t = np.arange(len(fit)) * dt
    axreal.plot(t, xbar, xc, label=r"$x_0$")
    axreal.plot(t, ybar, yc, label=r"$y_0$")
    axreal.set_xlabel('Time (s)')
    # add legend to real axis
    axreal.legend(loc='best')
    # calc FFTs
    Y = rfftn(ybar)
    X = rfftn(xbar)
    # calc FFT freq
    k = rfftfreq(len(fit), dt)
    # limit FFT display range
    kg = np.logical_and(k > lf, k < hf)
    # Plot FFT
    if log:
        axfft.semilogy(k[kg], abs(X[kg]), xc)
        axfft.semilogy(k[kg], abs(Y[kg]), yc)
    else:
        axfft.plot(k[kg], abs(X[kg]), xc)
        axfft.plot(k[kg], abs(Y[kg]), yc)
    axfft.set_xlabel('Frequency (Hz)')
    # Plot scatter
    axscatter.scatter(xbar, ybar, c=t, cmap=cmap)
    axscatter.set_xlabel('x')
    axscatter.set_ylabel('y')
    # tight layout
    axs = (axreal, axfft, axscatter)
    fig.tight_layout()
    # return fig, axs to user for further manipulation and/or saving if wanted
    return fig, axs


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
