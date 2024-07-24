# Version should always be readily available.
__version__ = '0.1.0'

# Lazy loading for sub-packages.
class _LazyLoader:
    def __init__(self, package_name):
        self._package_name = package_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = __import__(self._package_name, globals(), locals(), ['*'])
        return self._module

    def __getattr__(self, name):
        module = self._load()
        return getattr(module, name)

    def __dir__(self):
        module = self._load()
        return dir(module)

# Setup lazy loading for sub-packages.
features = _LazyLoader('mtflearn.features')
denoise = _LazyLoader('mtflearn.denoise')
io = _LazyLoader('mtflearn.io')
clustering = _LazyLoader('mtflearn.clustering')
utils = _LazyLoader('mtflearn.utils')



# Explicit imports for frequently used functions or classes
# These are assumed to be lightweight and commonly used enough to justify immediate loading.
from mtflearn.features._zps import ZPs   # Assuming this is lightweight
from mtflearn.features._zmoments import zmoments
from mtflearn.denoise._denoise_svd import denoise_svd
from mtflearn.denoise._denoise_svd import DenoiseSVD
from mtflearn.io._io_image import load_image

__all__ = ['features',
           'denoise',
           'io',
           'clustering',
           'utils',
           'ZPs',
           'zmoments',
           'denoise_svd',
           'DenoiseSVD',
           'load_image',
           ]