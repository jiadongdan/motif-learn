from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# colorbar
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def make_cbar_ax(ax, position='right', size=0.05, pad=0.05):
    x, y, w, h = ax.get_position().bounds

    if position == 'left':
        h = h
        w = w * size
        x = x - w - w / size * pad
        y = y
    elif position == 'right':
        h = h
        w = w * size
        x = x + w / size + w / size * pad
        y = y
    elif position == 'top':
        h = h * size
        w = w
        x = x
        y = y + h / size + h / size * pad
    elif position == 'bottom':
        h = h * size
        w = w
        x = x
        y = y - h / size * pad - h
    cax = ax.figure.add_axes([x, y, w, h])
    return cax

def make_cbar(ax, img=None, position='right', size=0.05, pad=0.05, cmap='viridis'):

    cax = make_cbar_ax(ax, position, size, pad)
    if img is None:
        mappable = ax.images[0]
    else:
        norm = Normalize(img.min(), img.max())
        mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar= ax.figure.colorbar(cax=cax)
    return cbar

def add_cbar(cax, vmin, vmax, cmap='viridis', **kwargs):
    norm = Normalize(vmin, vmax)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    cbar= cax.figure.colorbar(mappable=mappable, cax=cax, **kwargs)
    return cbar

def add_image_colorbar(ax, position='right', size=0.05, pad=0.05, **kwargs):
    # add an axis to place colorbar
    cax = make_cbar_ax(ax, position, size, pad)
    mappable = ax.images[0]
    if position in ['top', 'bottom']:
        orientation='horizontal'
    else:
        orientation='vertical'
    cbar = cax.figure.colorbar(mappable=mappable, cax=cax, orientation=orientation, **kwargs)
    return cbar
