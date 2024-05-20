from ._dm4 import DMfile
from skimage.io import imread

def normalize(image, low=0., high=1.):
    img_max = image.max()
    img_min = image.min()
    img_norm = (image - img_min) / (img_max - img_min) * (high - low) + low
    return img_norm


def load_image(file_name, normalized=False):
    file_extension = np.char.split(file_name, sep='.').tolist()[-1]
    file_extension = '.' + file_extension
    # file_extension = os.path.splitext(file_name)[1]
    if file_extension.lower() in ['.dm4', '.dm3']:
        img = DMfile(file_name).data
    else:
        img = imread(file_name)
    if normalized is True:
        img = normalize(img)
    return img