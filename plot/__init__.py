from .colors import generate_colors_from_lbs
from .colors import get_pca_colors
from ._color_data import cc
from .colors import to_rgb
from .colors import to_rgba
from .colors import to_hex
from .colors import colors_from_lbs
from .colors import Color
from .colors import Colors
from .colors import ColorTable

from .colormaps import color_palette
from .colormaps import color_mix
from .colormaps import show_cmap


from .layout1 import align_letters
from .layout1 import auto_letters
from .layout1 import auto_range
from .layout1 import auto_resize
from .layout1 import auto_square
from .layout1 import auto_tick_labels_size
from .layout1 import auto_tick_length
from .layout1 import ax_resize
from .layout1 import axes_from_ax
from .layout1 import connect_by_line
from .layout1 import draw_box
from .layout1 import get_ax_aspect
from .layout1 import get_ax_position
from .layout1 import get_fig_aspect
from .layout1 import get_position
from .layout1 import get_size_inches
from .layout1 import get_top_from_axes
from .layout1 import h_axes
from .layout1 import has_lbs
from .layout1 import init_letter_position
from .layout1 import make_cbar
from .layout1 import make_cbar_ax
from .layout1 import make_circular_axes
from .layout1 import make_inset_ax
from .layout1 import merge_axes
from .layout1 import shift_ax
from .layout1 import shift_axes


from .layout1 import ax_add_arrow
from .layout1 import fig_add_arrow
from .layout1 import make_ax_3d
from .layout1 import ax_zoom
from .layout1 import add_cbar
from .layout1 import make_bararrow
from .layout1 import ax_errorbar
from .layout1 import ax_boxplot


from .misc import plot_compare
from .misc import plot_image
from .misc import add_scale_bar
from .misc import add_anchor_text

from .misc import plot_pca_layout
from .misc import plot_FR_layout
from .misc import plot_xy
from .misc import plot_stacks

from .arrows import ax_add_arrow
from .arrows import fig_add_arrow
from .arrows import connect_by_arrow
from .arrows import align_arrows
from .arrows import shift_arrow
from .arrows import make_arrow
from .arrows import connect_by_fancyarrow
from .arrows import connect_by_arrow

from .data_slicer1 import plot
from .data_slicer1 import imshow
from .data_slicer1 import scatter
from .data_slicer import plot_zm
from .cursor import Cursor

#from .interactive_clusters_v4 import interactive_clusters
from .interactive_clusters_v7 import interactive_clusters


from .identification_paper import Feature_Vector
from .identification_paper import Feature_Matrix

from .ruler import Ruler

from .style import rc1, font_small, font_medium, font_large, font_letter, font_letter_nature
from .style import font_small_normal, font_medium_normal, font_large_normal


__all__ = ['Cursor',
 'Feature_Matrix',
 'Feature_Vector',
 'Ruler',
 'add_anchor_text',
 'add_scale_bar',
 'align_letters',
 'auto_letters',
 'auto_range',
 'auto_resize',
 'auto_square',
 'auto_tick_labels_size',
 'auto_tick_length',
 'ax_resize',
 'axes_from_ax',
 'cc',
 'color_mix',
 'color_palette',
 'colors_from_lbs',
 'connect_by_arrow',
 'connect_by_line',
 'draw_box',
 'font_large',
 'font_large_normal',
 'font_letter',
 'font_letter_nature',
 'font_medium',
 'font_medium_normal',
 'font_small',
 'font_small_normal',
 'generate_colors_from_lbs',
 'get_ax_aspect',
 'get_ax_position',
 'get_fig_aspect',
 'get_position',
 'get_pca_colors',
 'get_size_inches',
 'get_top_from_axes',
 'h_axes',
 'has_lbs',
 'imshow',
 'init_letter_position',
 'interactive_clusters',
 'make_cbar',
 'make_cbar_ax',
 'make_circular_axes',
 'make_inset_ax',
 'merge_axes',
 'plot',
 'plot_FR_layout',
 'plot_compare',
 'plot_image',
 'plot_pca_layout',
 'scatter',
 'plot_xy',
 'plot_zm',
 'rc1',
 'shift_ax',
 'shift_axes',
 'show_cmap',
 'to_hex',
 'to_rgb',
 'to_rgba',
 'make_ax_3d',
 'ax_add_arrow',
 'fig_add_arrow',
 'ax_zoom',
 'add_cbar',
 'plot_stacks',
 'ax_errorbar',
 'ax_boxplot',
 'align_arrows',
 'shift_arrow',
 'Colors',
 'Color',
 'ColorTable',
 'make_arrow',
 'connect_by_fancyarrow'
 ]