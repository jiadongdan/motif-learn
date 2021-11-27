import glob
import imageio
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity

from ..io.io_image import load_image

def make_gif_from_data(imgs, file_name=None, duration=None, cmap=None, loop=True):
    if cmap is None:
        cmap = plt.cm.viridis
    imgs_color = [cmap(rescale_intensity(img, out_range=(0., 1.))) for img in imgs]
    if file_name == None:
        file_name = 'animated.gif'
    if duration == None:
        duration = 0.5
    kwargs = {'duration': duration, 'loop': loop }
    imageio.mimsave(file_name, imgs_color, **kwargs)



def make_gif(file_name, fname=None, duration=1, loop=True):

    imgs = []
    for name in glob.glob(file_name):
        img = load_image(name)
        imgs.append(img)
    # duration in seconds
    kwargs = {'duration': duration, 'loop': loop }
    if fname is None:
        fname = 'animated.gif'
    imageio.mimsave(fname, imgs, **kwargs)