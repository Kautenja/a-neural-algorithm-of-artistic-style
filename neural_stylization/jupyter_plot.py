"""An animated plot for Jupyter notebook to visualize loop data realtime."""
import numpy as np
from matplotlib import pyplot as plt
from IPython import display


class JupyterPlot(object):
    """An animated plot for visualizing data in loops."""

    def __init__(self,
                 title: str='Title',
                 xlabel: str='X Value',
                 ylabel: str='Y Value'):
        """
        Initialize a new interative plot.

        Args:
            xlabel: the string to display on the x axis
            ylabel: the string to display on the y axis
        """
        if not isinstance(title, str):
            raise TypeError('title must be of type: str')
        if not isinstance(xlabel, str):
            raise TypeError('xlabel must be of type: str')
        if not isinstance(ylabel, str):
            raise TypeError('ylabel must be of type: str')
        # assign the instance variables
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        # initialize the list of fitnesses
        self.data = []

    def __call__(self, datum: list):
        """
        Update the plot with new data.

        Args:
            datum: the new data point to add and plot

        Returns: None
        """
        # append the data to the list
        self.data.append(datum)
        # plot the list of data
        plt.plot(self.data)
        # update the title of the plot
        plt.title(self.title)
        # update the x axis
        plt.xlabel(self.xlabel)
        # update the y axis
        plt.ylabel(self.ylabel)
        # clear the current output of the Jupyter notebook
        display.clear_output(wait=True)
        # show the new, current plot
        plt.show()


__all__ = ['JupyterPlot']
