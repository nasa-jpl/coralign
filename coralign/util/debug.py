"""Module to hold debugging functions (mainly for plotting images)."""
import matplotlib.pyplot as plt
from coralign.util.check import (twoD_array, real_array, real_scalar,
                                 string, boolean)


def debug_plot(debug, plot_num, img_arr, title):
    """
    Plot an image for visual debugging purposes.

    Parameters
    ----------
    debug : boolean
        True to plot. False to not plot.
    plot_num : float
        Plot number.
    img_arr : numpy ndarray
        2-D image to plot.
    title : string
        Plot title.
    """
    boolean(debug, 'debug', TypeError)
    twoD_array(img_arr, 'img_arr', TypeError)
    real_array(img_arr, 'img_arr', TypeError)
    real_scalar(plot_num, 'plot_num', TypeError)
    string(title, 'title', TypeError)

    if (debug):
        plt.figure(plot_num)
        plt.imshow(img_arr)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.pause(0.1)
