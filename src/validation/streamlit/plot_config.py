from cycler import cycler
from matplotlib import pyplot as plt

config = {'font.family' : 'Arial',
          'font.size'   : 7,
          'image.resample' : False,
          'figure.dpi' : 300,
          'axes.prop_cycle' : cycler('color', plt.get_cmap('Paired').colors),
          'xtick.major.size' : 2,
          'ytick.major.size' : 2,
          'xtick.major.width' : 0.5,
          'ytick.major.width' : 0.5,
          'axes.linewidth' : 0.5,
          'lines.linewidth' : 0.5}
