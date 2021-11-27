from scipy.signal import savgol_filter


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# 1D filter
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def savgol(x, window_length=5, polyorder=None):
    if window_length % 2 == 0:
        window_length = window_length + 1
    if polyorder is None:
        polyorder = 3
    return savgol_filter(x, window_length, polyorder=polyorder)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# 2D filter
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



