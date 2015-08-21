import numpy as np
from matplotlib import pyplot as plt

def display_grid(data, showcontour = False, contourcolor = 'w', figsize = 3, **kwargs):
    '''
    Display a dictionary of images in a nice grid

    Parameters
    ----------
    data : dict
        a dictionary of images
    showcontour: bool (default, False)
        Whether to show contours or not
    '''
    if not isinstance(data,dict):
        raise TypeError('Data is not a dictionary!')

    numitems = len(data)
    ncols = int(np.sqrt(numitems))
    nrows = int(np.ceil(numitems/ncols))

    fig, axs =  plt.subplots(nrows,ncols,figsize = (figsize*ncols,figsize*nrows))
    for (k, v), ax in zip(sorted(data.items()), axs.ravel()):
        ax.matshow(v,**kwargs)
        if showcontour:
            ax.contour(v,colors=contourcolor)
        ax.set_title(k)
        ax.axis('off')

    for ax in axs.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)

    return fig, axs
