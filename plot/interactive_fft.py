from matplotlib.widgets import RectangleSelector

class InteractiveFFT:

    def __init__(self, fig, img):

        self.img = img
        self.fig = fig
        self.subimg = None

        self.ax_img = None
        self.ax_fft = None

        self.selector = RectangleSelector(self.ax_img, onselect=self.onselect)


    def onselect(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
        print(f"The buttons you used were: {eclick.button} {erelease.button}")

