import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np


def setup(xtick_labelsize=8):

	# Set matplotlib backend.
	mpl.use('Agg')

	nice_fonts = {
        #"text.usetex": True,
        #"font.family": 'Libertine',
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        "lines.linewidth": 1.0,
        "axes.titlesize": 12,
        "lines.markersize": 3,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": 8
    }
	mpl.rcParams.update(nice_fonts)

        
def set_fig_size(width=None, height=None, fraction=1, subplots=(1, 1)):
    """ Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """

    if width is not None:
        
        if width == 'beamer':
            width_pt = 307.28987
        else:
            width_pt = width
        
        # Width of figure (in pts)
        fig_width_pt = width_pt * fraction

        # Convert from pt to inches
        inches_per_pt = 1 / 72.27

        # Golden ratio to set aesthetic figure height
        golden_ratio = (5 ** 0.5 - 1) / 2

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

        fig_dim = (fig_width_in, fig_height_in)

        return fig_dim


def set_arrowed_spines(fig, ax, eta=0.05, xshift=0, yshift=0):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1.0 / 40. * (ymax - ymin) 
    hl = 1.0 / 40. * (xmax - xmin)
    lw = 1 # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw / (ymax-ymin) * (xmax-xmin) * height / width 
    yhl = hl / (xmax-xmin) * (ymax-ymin) * width / height

    # draw x and y axis
    ax.arrow(xmin - xshift, ymin + yshift, xmax - xmin + eta * (xmax - xmin) - yshift, 0., fc='k', ec='k', lw=1, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on=False) 

    ax.arrow(xmin - xshift, ymin + yshift, 0.0, ymax - ymin + eta * (ymax - ymin) - yshift, fc='k', ec='k', lw=1, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on=False)


def set_ylim(values):

    diff = np.max(values) - np.min(values)

    if diff < 0.2:

        delta = 0.5 * (0.2 - diff)
        return [np.min(values) - delta, np.max(values) + delta]

    return None 


def format_axis(axis, fig, xtick_labels=None, x_values=None, xlim=None, ylim=None, n_xticks=6, xlabel=None, ylabel=None, 
                 grid=False, axis_title=None):

    if xtick_labels is not None:
        axis.set_xticks(range(len(xtick_labels)))
        axis.set_xticklabels(xtick_labels)

    if n_xticks > 0 and x_values is not None:

        axis.set_xticks(np.linspace(0, 1, n_xticks + 1))
        axis.set_xticklabels(np.round(np.linspace(min(x_values), max(x_values), n_xticks + 1), 3))

    axis.grid(linewidth=1, color="gray", alpha=0.2)

    if ylim is not None:
        axis.set_ylim(ylim[0], ylim[1])
    
    if xlim is not None:
        axis.set_xlim(xlim[0], xlim[1])

    if xlabel is not None:
        axis.set_xlabel(xlabel)

    if ylabel is not None:
        axis.set_ylabel(ylabel)

    if axis_title is not None:
        axis.set_title(axis_title)

    set_arrowed_spines(fig, axis)

    if grid:
        plt.grid(True)