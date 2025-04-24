from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt

def plot() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.grid()
    ax.tick_params(which='both', direction="in")

    return fig, ax

@dataclass
class AxesArray:
    array: Any
    ncols: int
    nrows: int
    
    def __getitem__(self, key) -> Axes:
        return self.array[key]
    
    def __iter__(self):
        return AxesIter(self, 0)
@dataclass
class AxesIter:
    axes: AxesArray
    current: int

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.axes.ncols * self.axes.nrows:
            raise StopIteration
        else:
            if self.axes.ncols == 1 or self.axes.nrows == 1:
                self.current += 1

                return self.axes[self.current - 1]
            else:
                j = self.current % self.axes.nrows
                i = self.current // self.axes.nrows

                self.current += 1
                return self.axes[i, j]

def plot_many(ncols: int, nrows: int, shape = None) -> tuple[Figure, AxesArray]:
    fig, axes = plt.subplots(ncols, nrows, figsize=shape)

    axes: AxesArray = AxesArray(axes, ncols, nrows)
    if ncols == 1 or nrows == 1:
        for i in range(ncols * nrows):
            axes[i].grid()
            axes[i].tick_params(which='both', direction="in")
    else:
        for i in range(ncols):
            for j in range(nrows):
                axes[i, j].grid()
                axes[i, j].tick_params(which='both', direction="in")

    return fig, axes

def plot_scattering(ax: Axes, x, y_re, y_im, label: str = ""):
    lines = ax.plot(x, y_re, label=label)
    ax.plot(x, y_im, linestyle="--", color=lines[0].get_color())

def load(filename: str, delimiter: str = '\t', skiprows:int = 1):
    data = np.loadtxt(f'{filename}', delimiter=delimiter, skiprows=skiprows)

    return data