from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import shutil

def plot() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")

    return fig, ax

def plot_scattering(ax: Axes, x, y_re, y_im, label: str = ""):
    lines = ax.plot(x, y_re, label=label)
    ax.plot(x, y_im, linestyle="--", color=lines[0].get_color())

def load(filename: str, delimiter: str = '\t', skiprows:int = 1):
    data = np.loadtxt(f'{filename}', delimiter=delimiter, skiprows=skiprows)

    return data