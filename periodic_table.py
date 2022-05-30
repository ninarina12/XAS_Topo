import numpy as np
import pandas as pd

from mendeleev.fetch import fetch_table
from bokeh.plotting import figure, output_file, save
from bokeh.io import export_png, export_svgs
from bokeh.models import ColumnDataSource, ColorBar, LogColorMapper, LogTicker, LinearColorMapper
from bokeh.core.properties import value

from selenium import webdriver
import geckodriver_autoinstaller
geckodriver_autoinstaller.install()

import matplotlib as mpl
import matplotlib.colors as colors


def colormap_column(df, column, cmap='RdBu_r', lognorm=False, cnorm=None, missing='#ffffff'):
    '''
    Return a new DataFrame with the same size (and index) as df with a column
    cmap containing HEX color mapping from cmap colormap.

    Args:
      df : Pandas DataFrame with the data
      column : Name of the column to be color mapped
      cmap : Name of the colormap, see matplotlib.org
      lognorm : True or False if logarithmic scale should be used for colormap
      cnorm : (minimum value, maximum value) for colormap normalization
      missing : HEX color for the missing values (NaN or None)
    '''

    colormap = mpl.cm.get_cmap(cmap)
    palette = [colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
    if lognorm:
        mapper = LogColorMapper
        normalizer = colors.LogNorm
    else:
        mapper = LinearColorMapper
        normalizer = colors.Normalize

    if cnorm:
        color_mapper = mapper(palette=palette, low=cnorm[0], high=cnorm[1])
        cnorm = normalizer(vmin=cnorm[0], vmax=cnorm[1])
    else:
        color_mapper = mapper(palette=palette, low=df[column].min(), high=df[column].max())
        cnorm = normalizer(vmin=df[column].min(), vmax=df[column].max())

    scalarmap = mpl.cm.ScalarMappable(norm=cnorm, cmap=colormap)
    out = pd.DataFrame(index=df.index)
    mask = df[column].isnull()
    rgba = scalarmap.to_rgba(df[column])
    out.loc[:, 'cmap'] = [colors.rgb2hex(row) for row in rgba]
    out.loc[mask, 'cmap'] = missing
    return out, color_mapper


def textcolor_column(df, column, color='black', missing='#a9a9a9'):
    '''
    Return a new DataFrame with the same size (and index) as df with a column
    text_color containing text color preferences.

    Args:
      df : Pandas DataFrame with the data
      column : Name of the column to be color mapped
      color : Name of the text color
      missing : HEX color for the missing values (NaN or None)
    '''

    out = pd.DataFrame(index=df.index)
    mask = df[column].isnull()
    out.loc[:, 'text_color'] = color
    out.loc[mask, 'text_color'] = missing
    return out


def periodic_plot(df, attribute='atomic_weight', title='',
                  width=750, height=550, missing='#ffffff', decimals=2,
                  colorby=None, output=None, cmap='RdBu_r', lognorm=False,
                  cnorm=None, showfblock=True, long_version=False):
    '''
    Use Bokeh to plot the periodic table data contained in the df.

    Args:
      df : Pandas DataFrame with the data on elements
      attribute : Name of the attribute to be displayed
      title : Title to appear above the periodic table
      colorby : Name of the column containig the colors
      width : Width of the figure in pixels
      height : Height of the figure in pixels
      decimals : Number of decimals to be displayed in the bottom row of each cell
      missing : Hex code of the color to be used for the missing values
      output : Name of the output file to store the plot, should end in .html
      cmap : Colormap to use, see matplotlib colormaps
      lognorm : True or False if logarithmic scale should be used for colormap
      cnorm : (minimum value, maximum value) for colormap normalization
      long_version : Show the long version of the periodic table with the f block between the s and d blocks
      showfblock : Show the elements from the f block
    '''

    ac = 'attribute_column'

    elements = df.copy()

    # calculate x and y of the main group/row elements
    elements.loc[elements['group_id'].notnull(), 'x'] = \
        elements.loc[elements['group_id'].notnull(), 'group_id'].astype(int)
    elements.loc[elements['period'].notnull(), 'y'] = \
        elements.loc[elements['period'].notnull(), 'period'].astype(int)

    if showfblock:
        if long_version:
            elements.loc[elements['x'] > 2, 'x'] = \
                elements.loc[elements['x'] > 2, 'x'] + 14
            for period in [6, 7]:
                mask = (elements['block'] == 'f') & (elements['period'] == period)
                elements.loc[mask, 'x'] = elements.loc[mask, 'atomic_number'] -\
                                             elements.loc[mask, 'atomic_number'].min() + 3
                elements.loc[mask, 'y'] = period
        else:
            for period in [6, 7]:
                mask = (elements['block'] == 'f') & (elements['period'] == period)
                elements.loc[mask, 'x'] = elements.loc[mask, 'atomic_number'] -\
                                             elements.loc[mask, 'atomic_number'].min() + 3
                elements.loc[mask, 'y'] = elements.loc[mask, 'period'] + 2

    # additional columns for positioning of the text
    elements.loc[:, 'y_anumber'] = elements['y'] - 0.3
    elements.loc[:, 'y_name'] = elements['y'] + 0.2
    elements.loc[:, 'y_symbol'] = elements['y'] - 0.1
    
    if property:
        elements.loc[elements[attribute].notnull(), 'y_prop'] = elements.loc[elements[attribute].notnull(), 'y'] + 0.3
    else:
        elements.loc[:, 'y_prop'] = elements['y'] + 0.3

    temp = textcolor_column(elements, attribute)
    elements = pd.merge(elements, temp, left_index=True, right_index=True)

    if colorby == 'attribute':
        temp, color_mapper = colormap_column(elements, attribute, cmap=cmap, lognorm=lognorm,
                                             cnorm=cnorm, missing=missing)
        elements = pd.merge(elements, temp, left_index=True, right_index=True)
        colorby = 'cmap'

    if elements[attribute].dtype == np.float64:
        elements[ac] = elements[attribute].round(decimals=decimals)
    else:
        elements[ac] = elements[attribute]
    
    if colorby not in elements.columns:
        series = fetch_table('series')
        elements = pd.merge(elements, series, left_on='series_id',
                            right_on='id', suffixes=('', '_series'))
        colorby = 'color'

    source = ColumnDataSource(data=elements)

    TOOLS = "save,reset"

    p = figure(title=title,
           tools=TOOLS,
           x_axis_location='above',
           x_range=(elements.x.min() - 0.5, elements.x.max() + 0.5),
           y_range=(elements.y.max() + 0.5, elements.y.min() - 0.5),
           width=width, height=height,
           toolbar_location='above',
           toolbar_sticky=False,
           )

    p.rect("x", "y", 0.9, 0.9, source=source, color=colorby, fill_alpha=0.7)
    
    # add colorbar
    cbar_props = {
        "label_standoff": 2,
        "location": (0,228),
        "height": 500,
        "border_line_color": None,
        "major_label_text_font_size": "14pt",
        "major_label_text_font": "Myriad Pro",
        "major_label_text_align": "left"
    }
    if lognorm:
        color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(), **cbar_props)
    else:
        color_bar = ColorBar(color_mapper=color_mapper, **cbar_props)

    #p.add_layout(color_bar, 'right')
    
    # adjust the ticks and axis bounds
    p.yaxis.bounds = (1, 7)
    p.axis.visible = False
    
    text_props = {
        "source": source,
        "angle": 0,
        "text_align": "center",
        "text_baseline": "middle",
        "text_font": value("Myriad Pro")
    }

    p.title.text_font = "Myriad Pro"

    p.text(x="x", y="y_symbol", text="symbol", text_font_size="16pt", text_color="text_color", **text_props)
    
    p.text(x="x", y="y_prop", text=ac,
           text_font_size="12pt", text_color="black", **text_props)

    p.grid.grid_line_color = None

    if output:
        p.output_backend = "svg"
        export_svgs(p, filename=output + '.svg')
