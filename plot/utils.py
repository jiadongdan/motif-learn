
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# bounding box
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def ax_get_position(ax, loc=None, pad=0.005, transform='fig'):
    fig = ax.figure
    # fig fraction coordinates
    if transform == 'fig':
        x, y, w, h = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).bounds
    # pixel coordinates
    elif transform in ['display', 'pixel']:
        x, y, w, h = ax.get_tightbbox(fig.canvas.get_renderer()).bounds
        # convert to pixel space
        fig2pixel = ax.figure.transFigure
        pad = fig2pixel.transform([pad, pad])[0]

    if loc == 'left':
        return (x-pad, y + h / 2, w, h)
    elif loc == 'right':
        return (x + w + pad, y + h / 2, w, h)
    elif loc == 'bottom':
        return (x + w / 2, y-pad, w, h)
    elif loc == 'top':
        return (x + w / 2, y + h+pad, w, h)
    elif loc == 'center':
        return (x + w / 2, y + h / 2, w, h)
    else:
        return (x, y, w, h)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# aspect ratio
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def get_size_inches(ax):
    fig_w, fig_h = ax.figure.get_size_inches()
    w, h = ax.get_position().bounds[2:]
    return (fig_w * w, fig_h * h)


def get_ax_aspect(ax):
    w, h = get_size_inches(ax)
    return w / h


def get_fig_aspect(fig):
    fig_w, fig_h = fig.get_size_inches()
    return fig_w / fig_h

