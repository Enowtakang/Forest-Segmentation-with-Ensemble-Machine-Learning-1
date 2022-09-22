#   https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec?
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
                Remember to adapt the grid
                at line 63, when using a different
                dataset
"""

"""
Load data
"""
data = pd.read_csv('features_segment_multiple.csv')
# Remove space in column names: https://www.geeksforgeeks.org/remove-spaces-from-column-names-in-pandas/
data.columns = data.columns.str.replace(' ', '')


"""
Intro code
"""
# Use 256 colors for the diverging color palette
n_colors = 256

# Create the palette
palette = sns.diverging_palette(20, 220, n=n_colors)

# Range of values that will be mapped to the palette,
# i.e. min and max possible correlation
color_min, color_max = [-1, 1]


def value_to_color(val):
    # position of value in the input range,
    # relative to the length of the input range
    val_position = float((val - color_min)) / (color_max - color_min)
    # target index in the color palette
    ind = int(val_position * (n_colors - 1))
    return palette[ind]


"""
Step 1 - Make a scatter plot with square markers, 
set column names as labels
"""


def heat_map(x, y, size,
             color
             # https://medium.com/@dgkadesewa/thanks-for-this-i-tried-to-use-your-code-but-got-an-error-here-6deb35eb3b8e
             ):
    fig, ax = plt.subplots()

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    # Setup a 1x21 grid
    plot_grid = plt.GridSpec(1, 21, hspace=0.2, wspace=0.1)
    # Use the leftmost 20 columns of the grid
    # for the main plot
    ax = plt.subplot(plot_grid[:, :-1])

    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        c=color.apply(value_to_color),  # Vector of square color values, mapped to color palette
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels,
                       rotation=90,
                       horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks(
        [t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks(
        [t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max(
        [v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max(
        [v for v in y_to_num.values()]) + 0.5])

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )

    ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False)  # Hide grid
    ax.set_facecolor('white')  # Make background white
    ax.set_xticks([])  # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()  # Show vertical ticks on the right


"""
Make Correlation
"""
corr = data.corr()

"""
Unpivot the dataframe, so we can get pair 
of arrays for x and y
"""
corr = pd.melt(
    corr.reset_index(),
    id_vars='index')
corr.columns = ['x', 'y', 'value']

heat_map(
    x=corr['x'], y=corr['y'],
    size=corr['value'].abs(),
    color=corr['value'])

plt.show()


