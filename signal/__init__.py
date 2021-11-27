from .baseline import baseline_correction

from .filters import savgol

from .peak_finding import get_patch_size
from .peak_finding import local_max

__all__ = ['baseline_correction',
           'local_max',
           'get_patch_size',
           'savgol'
           ]