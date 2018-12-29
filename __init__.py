# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:31:00 2015

@author: david
"""

import numpy as np
import warnings
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Colormap, LogNorm, PowerNorm
import matplotlib.font_manager as fm
import mpl_toolkits.axes_grid1.inset_locator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import mpl_toolkits.axes_grid1 as mpl
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


# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''

# Data manipulation:


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap='inferno', norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, ax=None, autoscale=True):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    if not isinstance(cmap, Colormap):
        cmap = plt.get_cmap(cmap)

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    
    lc.set_capstyle("round")

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    if autoscale:
        ax.autoscale()
    
    return lc


def display_grid(data, showcontour=False, contourcolor='w', filter_size=None,
                 figsize=3, auto=False, nrows=None, grid_aspect=None, sharex=False, sharey=False,
                 **kwargs):
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
    # figure out grid_aspect ratios of data (a = y / x)
    if grid_aspect is None:
        aspects = np.array([
            v.shape[0] / v.shape[1] for v in data.values() if v.ndim > 1
        ])
        # if len is zero then everything was 1d
        if len(aspects):
            grid_aspect = aspects.mean()
            if not np.isfinite(grid_aspect):
                raise RuntimeError(
                    "grid_aspect isn't finite, grid_aspect = {}".format(grid_aspect))
        else:
            grid_aspect = 1
    fig, axs = make_grid(len(data), nrows=nrows, figsize=figsize,
                         grid_aspect=grid_aspect, sharex=sharex, sharey=sharey)
    for (k, v), ax in zip(sorted(data.items()), axs.ravel()):
        if v.ndim == 1:
            ax.plot(v, **kwargs)
        else:
            if auto:
                # calculate vmin, vmax
                kwargs.update(auto_adjust(v))
            ax.matshow(v, **kwargs)
            if showcontour:
                if filter_size is None:
                    vv = v
                else:
                    vv = fft_gaussian_filter(v, filter_size)

                ax.contour(vv, colors=contourcolor)
        ax.set_title(k)
        ax.axis('off')

    clean_grid(fig, axs)

    return fig, axs


def make_grid(numitems, nrows=None, figsize=3, grid_aspect=1, **kwargs):
    if numitems == 0:
        raise ValueError("numitems can't be zero.")
    if nrows is None:
        nrows = int(np.sqrt(numitems))
    if nrows == 0:
        nrows = ncols = 1
    else:
        ncols = int(np.ceil(numitems / nrows))

    fig, axs = plt.subplots(
        nrows, ncols, figsize=(figsize * ncols, figsize * nrows * grid_aspect),
        squeeze=False, **kwargs
    )

    return fig, axs


def clean_grid(fig, axs):
    for ax in axs.ravel():
        if not (len(ax.images) or len(ax.lines) or len(ax.patches)):
            fig.delaxes(ax)
    return fig, axs


def take_slice(data, axis, midpoint=None):
    """Small utility to be able to take slices"""
    if midpoint is None:
        midpoint = np.array(data.shape, dtype=np.int) // 2
    my_slice = [slice(None, None, None) for i in range(data.ndim)]
    my_slice[axis] = midpoint[axis]
    return data[my_slice]


def slice_plot(data, center=None, allaxes=False, **kwargs):
    """A slice plot, displays slices through data at `center`"""
    take_slice2 = partial(take_slice, midpoint=center)
    return mip(data, func=take_slice2, allaxes=allaxes, **kwargs)


def recolor(cmap, ax=None, new_alpha=None, to_change='lines'):
    '''
    Recolor the lines in ax with the cmap
    '''
    if isinstance(cmap, str):
        # user has passed a string
        # presumably the name of a registered color map
        cmap = plt.get_cmap(cmap)

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
        new_color = list(cmap(i / (num_objs - 1)))
        # replace alpha is wanted
        if new_alpha is not None:
            new_color[-1] = new_alpha
        # set the color
        obj.set_color(new_color)


def drift_plot(fit, title=None, dt=0.1, dx=130, lf=-np.inf, hf=np.inf,
               log=False, cmap='magma', xc='b', yc='r'):
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
    dx : pixel size (optional)
        Pixel size in nm
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
    ybar *= dx
    xbar = fit.x0 - fit.x0.mean()
    xbar *= dx
    # Plot Real space
    t = np.arange(len(fit)) * dt
    axreal.plot(t, xbar, xc, label=r"$x_0$")
    axreal.plot(t, ybar, yc, label=r"$y_0$")
    axreal.set_xlabel('Time (s)')
    axreal.set_ylabel('Displacement (nm)')
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
    # make sure the scatter plot is square
    lims = axreal.get_ylim()
    axscatter.set_ylim(lims)
    axscatter.set_xlim(lims)
    # tight layout
    axs = (axreal, axfft, axscatter)
    fig.tight_layout()
    # return fig, axs to user for further manipulation and/or saving if wanted
    return fig, axs


def mip(data, zaspect=1, func=np.amax, allaxes=False, plt_kwds=None, **kwargs):
    """
    Plot max projection of data

    Parameters
    ----------------
    data : 2 or 3 dimensional ndarray
        the data to be plotted
    func :  callable
        a function to be called on the data, must accept and axes argument
    allaxes : bool
        whether to return all axes or not
    plt_kwds : dict
        A dictionary of keywords for the plots (2D case)
    kwargs : dict
        passed to matshow

    Returns
    ----------------
    fig : mpl figure instanse
        figure handle
    axs : ndarray of axes objects
        axes handles in a flat ndarray
    """

    # set default properly for dict
    if plt_kwds is None:
        plt_kwds = {}
    # pull shape and dimensions
    myshape = data.shape
    ndim = data.ndim
    # set up the grid for the subplots
    if ndim == 3:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, myshape[0] * zaspect / myshape[2]],
                               height_ratios=[1, myshape[0] * zaspect / myshape[1]])
        # set up my canvas necessary to make the overall figure shape square,
        # without this the boxes aren't sized properly
        fig = plt.figure(figsize=(5 * myshape[2] / myshape[1], 5))
    elif ndim == 2:
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
        fig = plt.figure(figsize=(5, 5))
    else:
        raise TypeError("Data has too many dimensions, ndim = {}".format(ndim))
    # set up each projection
    ax_xy = plt.subplot(gs[0])
    ax_xy.set_title('XY')
    # set up YZ
    ax_yz = plt.subplot(gs[1], sharey=ax_xy)
    ax_yz.set_title('YZ')
    # set up XZ
    ax_xz = plt.subplot(gs[2], sharex=ax_xy)
    ax_xz.set_title('XZ')
    # actually calc data and plot
    if ndim == 3:
        max_z = func(data, axis=0)
        ax_xy.matshow(max_z, **kwargs)
        max_y = func(data, axis=1)
        ax_xz.matshow(max_y, aspect=zaspect, **kwargs)
        max_x = func(data, axis=2)
        ax_yz.matshow(max_x.T, aspect=zaspect, **kwargs)
    else:
        max_z = data.copy()
        ax_xy.matshow(max_z, **kwargs)
        max_x = func(data, axis=1)
        ax_yz.plot(max_x, np.arange(myshape[0]), **plt_kwds)
        max_y = func(data, axis=0)
        ax_xz.plot(max_y, **plt_kwds)
    for ax in (ax_xy, ax_xz, ax_yz):
        ax.axis("off")
    # make all axis tight
    for ax in (ax_xy, ax_yz, ax_xz):
        ax.axis("tight")
    fig.tight_layout()
    # if the user requests all axes return them
    if allaxes:
        return fig, np.array([ax_xy, ax_yz, ax_xz, plt.subplot(gs[3])])
    else:
        return fig, np.array([ax_xy, ax_yz, ax_xz])


def auto_adjust(img):
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
    limit = pixel_count / 10
    # histogram
    try:
        my_hist, bins = np.histogram(np.nan_to_num(img).ravel(), bins="auto")
        if len(bins) < 100:
            my_hist, bins = np.histogram(np.nan_to_num(img).ravel(), bins=128)
        # convert bin edges to bin centers
        bins = np.diff(bins) + bins[:-1]
        nbins = len(my_hist)
        # set up the threshold
        # Below is what ImageJ purportedly does.
        # auto_threshold = threshold_isodata(img, nbins=bins)
        # if auto_threshold < 10:
        #     auto_threshold = 5000
        # else:
        #     auto_threshold /= 2
        # this version, below, seems to converge as nbins increases.
        threshold = pixel_count / (nbins * 16)
        # find the minimum by iterating through the histogram
        # which has 256 bins
        valid_bins = bins[np.logical_and(my_hist < limit, my_hist > threshold)]
        # check if the found limits are valid.
        vmin = valid_bins[0]
        vmax = valid_bins[-1]
    except IndexError:
        vmin = 0
        vmax = 0

    if vmax <= vmin:
        vmin = img.min()
        vmax = img.max()

    return dict(vmin=vmin, vmax=vmax)


# @np.vectorize
def wavelength_to_rgb(wavelength, gamma=0.8):
    """This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    """
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B)


def max_min(n, d):
    return np.array((-n // 2, (n - 1) // 2 + n % 2)) * d


def fft_max_min(n, d):
    step_size = 1 / d / n
    return max_min(n, step_size)


def add_scalebar(ax, scalebar_size, pixel_size, unit="µm", **kwargs):
    """Add a scalebar to the axis"""
    scalebar_length = scalebar_size / pixel_size
    default_scale_bar_kwargs = dict(
        loc='lower right',
        pad=0.5,
        color='white',
        frameon=False,
        size_vertical=scalebar_length / 10,
        fontproperties=fm.FontProperties(size="large", weight="bold")
    )
    if unit is not None:
        label = '{} {}'.format(scalebar_size, unit)
    else:
        label = ""
    default_scale_bar_kwargs.update(kwargs)
    scalebar = AnchoredSizeBar(ax.transData,
                               scalebar_length,
                               label,
                               **default_scale_bar_kwargs
                               )
    # add the scalebar
    ax.add_artist(scalebar)


def z_squeeze(n1, n2, NA=0.85):
    """Amount z expands or contracts when using an objective designed
    for one index (n1) to image into a medium with another index (n2)"""
    def func(n):
        return n - np.sqrt(n**2 - NA**2)
    return func(n1) / func(n2)


def psf_plot(psf, NA=0.85, nobj=1.0, nsample=1.3, zstep=0.25, pixel_size=0.13,
             fig=None, loc=111, **kwargs):
    # expand z step
    zstep *= z_squeeze(nobj, nsample, NA)
    # update our default kwargs for plotting
    dkwargs = dict(norm=LogNorm(), interpolation="nearest", cmap="inferno")
    dkwargs.update(kwargs)
    # make the fig if one isn't passed
    if fig is None:
        fig = plt.figure(None, (8., 8.))

    grid = mpl.ImageGrid(fig, loc,
                         nrows_ncols=(2, 2),
                         axes_pad=0.3,
                         )
    # calc extents
    nz, ny, nx = psf.shape
    kz, ky, kx = [max_min(n, d) for n, d in zip(psf.shape, (zstep, pixel_size, pixel_size))]

    # do plotting
    grid[3].imshow(psf.max(0), **dkwargs, extent=(*kx, *ky))
    grid[2].imshow(psf.max(1).T, **dkwargs, extent=(*kz, *ky))
    grid[1].imshow(psf.max(2), **dkwargs, extent=(*kx, *kz))
    grid[0].remove()

    fd = {'fontsize': 16,
          'fontweight': "bold"}
    # add titles
    grid[3].set_title("$XY$", fd)
    grid[2].set_title("$YZ$", fd)
    grid[1].set_title("$XZ$", fd)
    # remove ticks
    for g in grid:
        g.set_xticks([])
        g.set_yticks([])
    # add scalebar
    add_scalebar(grid[3], 1)
    # return fig and axes
    return fig, grid


def otf_plot(otf, NA=0.85, wl=0.52, nobj=1.0, nsample=1.3, zstep=0.25, pixel_size=0.13, fig=None, loc=111, **kwargs):

    # expand z step
    zstep *= z_squeeze(nobj, nsample, NA)
    # update our default kwargs for plotting
    dkwargs = dict(norm=LogNorm(), interpolation="nearest", cmap="inferno")
    dkwargs.update(kwargs)
    # make the fig if one isn't passed
    if fig is None:
        fig = plt.figure(None, (8., 8.))

    grid = mpl.ImageGrid(fig, loc,
                         nrows_ncols=(2, 2),
                         axes_pad=0.3,
                         )
    
    nz, ny, nx = otf.shape
    kz, ky, kx = [fft_max_min(n, d) for n, d in zip(otf.shape, (zstep, pixel_size, pixel_size))]
    
    grid[3].imshow(otf[nz // 2, :, :], **dkwargs, extent=(*kx, *ky))
    grid[2].imshow(otf[:, ny // 2, :].T, **dkwargs, extent=(*kz, *ky))
    grid[1].imshow(otf[:, :, nx // 2], **dkwargs, extent=(*kx, *kz))
    grid[0].remove()
    fd = {'fontsize': 16,
          'fontweight': "bold"}
    grid[3].set_title("$k_{XY}$", fd)
    grid[2].set_title("$k_{YZ}$", fd)
    grid[1].set_title("$k_{XZ}$", fd)
    
    for g in grid:
        g.set_xticks([])
        g.set_yticks([])

    # calculate the angle of the marginal rays
    a = np.arcsin(NA / nsample)
    # make a circle of the OTF limits
    c = patches.Circle((0, 0), 2 * NA / wl, ec="w", lw=2, fill=None)
    grid[3].add_patch(c)
    # add bowties
    n_l = nsample / wl
    for b, g in zip((0, np.pi / 2), grid[1:3]):
        for j in (0, np.pi):
            for i in (0, np.pi):
                c2 = patches.Wedge((n_l * np.sin(a + b + j), n_l * np.cos(a + b + i)), n_l,
                                   np.rad2deg(-a - np.pi / 2 + i * np.cos(b) - (j + np.pi) * np.sin(b) + b),
                                   np.rad2deg(a - np.pi / 2 + i * np.cos(b) - (j + np.pi) * np.sin(b) + b),
                                   width=0, ec="w", lw=1, fill=None)
                g.add_patch(c2)
    # add scalebar
    add_scalebar(grid[3], 1, "µm$^{-1}$")

    return fig, grid