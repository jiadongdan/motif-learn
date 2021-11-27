import numpy as np


class Tooltip:

    def __init__(self, sc):
        self.sc = sc
        self.ax = self.sc.axes
        self.fig = self.ax.figure


        self.press = self.fig.canvas.mpl_connect("pcik_event", self.onpick)


    def onpick(self, event):
        print(event.artist)




