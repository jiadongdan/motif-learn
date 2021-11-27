import numpy as np
from matplotlib.widgets import AxesWidget


class Cursor(AxesWidget):
    """
    A horizontal and vertical line that spans the axes and moves with
    the pointer.  You can turn off the hline or vline respectively with
    the following attributes:

      *horizOn*
        Controls the visibility of the horizontal line

      *vertOn*
        Controls the visibility of the horizontal line

    and the visibility of the cursor itself with the *visible* attribute.

    For the cursor to remain responsive you must keep a reference to
    it.
    """

    def __init__(self, ax, horizOn=True, vertOn=True, useblit=True, transform=None,
                 **lineprops):
        """
        Add a cursor to *ax*.  If ``useblit=True``, use the backend-dependent
        blitting features for faster updates.  *lineprops* is a dictionary of
        line properties.
        """
        AxesWidget.__init__(self, ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)
        self.connect_event('button_press_event', self.onclick)

        self.visible = True
        self.horizOn = horizOn
        self.vertOn = vertOn
        self.useblit = useblit and self.canvas.supports_blit
        self.ax = ax
        if transform is None:
            self.transform = 'axes'
        else:
            self.transform = transform

        if self.useblit:
            lineprops['animated'] = True
        self.lineh = ax.axhline(ax.get_ybound()[0], visible=False, **lineprops)
        self.linev = ax.axvline(ax.get_xbound()[0], visible=False, **lineprops)

        self.background = None
        self.needclear = False

    def clear(self, event):
        """clear the cursor"""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.linev.set_visible(False)
        self.lineh.set_visible(False)

    def onmove(self, event):
        """on mouse motion draw the cursor if visible"""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.linev.set_visible(False)
            self.lineh.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return
        self.linev.set_xdata((event.xdata, event.xdata))

        self.lineh.set_ydata((event.ydata, event.ydata))
        self.linev.set_visible(self.visible and self.vertOn)
        self.lineh.set_visible(self.visible and self.horizOn)

        self._update()

    def onclick(self, event):
        x, y = event.xdata, event.ydata
        # transform from data coords to axes coords
        # https://stackoverflow.com/a/62004544/5855131
        if self.transform == 'axes':
            xy1 = self.ax.transData.transform((x, y))
            x, y = self.ax.transAxes.inverted().transform(xy1)

        ss = "ax.text(x={:.4f}, y={:.4f}, s='text', ha='left', va='top', transform=ax.transAxes)".format(x, y)
        cc = "c = plt.Circle(({:.4f}, {:.4f}), radius=0.02, fill=False, transform=ax.transAxes)".format(x, y)
        tt = 'ax.add_patch(c)'
        print(ss)
        print(cc)
        print(tt)

    def _update(self):

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.linev)
            self.ax.draw_artist(self.lineh)
            self.canvas.blit(self.ax.bbox)
        else:

            self.canvas.draw_idle()

        return False



class LineCursor(AxesWidget):

    def __init__(self, ax, slope=0, angle=0, useblit=True, transform=None, **lineprops):

        AxesWidget.__init__(self, ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)
        self.connect_event('button_press_event', self.onclick)

        self.visible = True
        if angle is not None:
            self.slope = np.tan(np.deg2rad(angle))
        else:
            self.slope = slope

        self.useblit = useblit and self.canvas.supports_blit
        self.ax = ax
        if transform is None:
            self.transform = 'axes'
        else:
            self.transform = transform

        if self.useblit:
            lineprops['animated'] = True
        self.line = ax.axline(ax.get_xbound(), slope=self.slope, visible=False, **lineprops)

        self.background = None
        self.needclear = False

    def clear(self, event):
        """clear the cursor"""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.line.set_visible(False)

    def onmove(self, event):
        def _to_points(xy1, xy2, slope):
            """
            Check for a valid combination of input parameters and convert
            to two points, if necessary.
            """
            if (xy2 is None and slope is None or
                    xy2 is not None and slope is not None):
                raise TypeError(
                    "Exactly one of 'xy2' and 'slope' must be given")
            if xy2 is None:
                x1, y1 = xy1
                xy2 = (x1, y1 + 1) if np.isinf(slope) else (x1 + 1, y1 + slope)
            return xy1, xy2

        """on mouse motion draw the cursor if visible"""
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.line.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return
        # change here
        xy1 = (event.xdata, event.ydata)
        xy2 = None
        datalim = [xy1] if xy2 is None else [xy1, xy2]
        (x1, y1), (x2, y2) = _to_points(xy1, xy2, self.slope)

        self.line.set_xdata([x1, x2])
        self.line.set_ydata([y1, y2])
        self.line.set_visible(self.visible)
        self._update()

    def onclick(self, event):
        x, y = event.xdata, event.ydata
        print(x, y)
        # transform from data coords to axes coords
        # https://stackoverflow.com/a/62004544/5855131
        if self.transform == 'axes':
            xy1 = self.ax.transData.transform((x, y))
            x, y = self.ax.transAxes.inverted().transform(xy1)

        ss = "ax.text(x={:.4f}, y={:.4f}, s='text', ha='left', va='top', transform=ax.transAxes)".format(x, y)
        cc = "c = plt.Circle(({:.4f}, {:.4f}), radius=0.02, fill=False, transform=ax.transAxes)".format(x, y)
        tt = 'ax.add_patch(c)'
        print(ss)
        print(cc)
        print(tt)

    def _update(self):

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.line)
            self.canvas.blit(self.ax.bbox)
        else:

            self.canvas.draw_idle()

        return False


from matplotlib.widgets import AxesWidget
from matplotlib.patches import Circle

class CircleCursor(AxesWidget):

    def __init__(self, ax, useblit=True, **kwargs):

        AxesWidget.__init__(self, ax)

        self.connect_event('motion_notify_event', self.onmove)
        self.connect_event('draw_event', self.clear)

        self.useblit = useblit and self.canvas.supports_blit
        self.ax = ax

        self.visible = True
        self.background = None
        self.needclear = False

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        self.x0, self.y0 = (xmin+xmax)/2, (ymin+ymax)/2

        self.circle = Circle(xy=(self.x0, self.y0), radius=0, fill=False)

    def clear(self, event):
        """clear the cursor"""
        if self.ignore(event):
            return
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.circle.set_visible(False)

    def onmove(self, event):
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return
        if event.inaxes != self.ax:
            self.circle.set_visible(False)

            if self.needclear:
                self.canvas.draw()
                self.needclear = False
            return
        self.needclear = True
        if not self.visible:
            return

        x, y = (event.xdata, event.ydata)
        r = np.hypot(x-self.x0, y-self.y0)
        self.circle.set_radius(r)
        self._update()


    def _update(self):

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.circle)
            self.canvas.blit(self.ax.bbox)
        else:

            self.canvas.draw_idle()

        return False