import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.font_manager as font_manager

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

font_path = '/usr/share/fonts/truetype/lato/Lato-Semibold.ttf'
prop = font_manager.FontProperties(fname=font_path)
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['mathtext.default'] = 'regular'

palette = ['#E7E6E6', '#E69A9C', '#FFE699', '#97B7C2', '#6B6AA8']
palette3 = ['#6A71B4', '#BB6D89', '#F7B744']
cmap3 = colors.LinearSegmentedColormap.from_list('cmap3', palette3, N=100)
cmap = colors.LinearSegmentedColormap.from_list('cmap', palette, N=500)
cmap_div = colors.LinearSegmentedColormap.from_list('cmap_div', ['#6A71B4', '#F7B744'], N=100)
cmap_div2 = colors.LinearSegmentedColormap.from_list('cmap_div', ['#6A71B4', '#E7E6E6', '#F7B744'], N=100)

# plotting format
def format_axis(ax, xlabel, ylabel, prop, title=None, legend=None):
    if title: ax.set_title(title, fontproperties=prop)

    ax.set_xlabel(xlabel, fontproperties=prop)
    ax.set_ylabel(ylabel, fontproperties=prop)
    ax.tick_params(direction='in', length=6, width=1)
    ax.tick_params(which='minor', direction='in')

    for lab in ax.get_xticklabels():
        lab.set_fontproperties(prop)
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(prop)
    
    if legend: ax.legend(frameon=False, prop=prop)

# some utilities for custom colormaps
def hex_to_rgb(color):
    color = color.strip("#") # remove the # symbol
    n = len(color)
    return tuple(int(color[i:i+n//3],16) for i in range(0,n,n//3))

def rgb_to_dec(color):
    return [c/256. for c in color]

# make a custom colormap given a list of hex colors and corresponding float values
def make_continuous_cmap(hex_colors, float_values):
    n = len(hex_colors) # should be the same as float_values
    rgb_colors = [rgb_to_dec(hex_to_rgb(color)) for color in hex_colors]
    cdict = {}
    primary = ['red','green','blue']
    for i,p in enumerate(primary):
        cdict[p] = [[float_values[j], rgb_colors[j][i], rgb_colors[j][i]] for j in range(n)]
        cmap = colors.LinearSegmentedColormap('my_cmap', segmentdata=cdict, N=500)
    return cmap

# make a monochromatic colormap that varies in opacity
def make_transparent_cmap(hex_color, alpha_lo, alpha_hi):
    cmap = colors.LinearSegmentedColormap.from_list('my_cmap',2*[hex_color],500)
    cmap._init()
    alphas = np.linspace(alpha_lo,alpha_hi,cmap.N+3)
    cmap._lut[:,-1] = alphas
    return cmap

# plot a colormap
def plot_cmap(cmap, ticks, savename):
    N = 100; r = 10
    m = ticks[-1]
    fig = Figure(figsize=(4,4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.imshow(np.linspace(0,1,N).reshape(1,-1), cmap=cmap, aspect=r)
    ax.set_yticks([])
    ax.set_xticks([t/m*N for t in ticks])
    ax.set_xticklabels(ticks)
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(prop)

    fig.tight_layout()
    fig.savefig(savename)

