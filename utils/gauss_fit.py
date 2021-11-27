# http://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
# http://scipy-cookbook.readthedocs.io/items/FittingData.html

import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize


def twodgaussian(amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    theta = float(theta)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    return lambda x, y: offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                                       + c * ((y - yo) ** 2)))


def moments(data):
    total = np.abs(data).sum()
    Y, X = np.indices(data.shape)
    # first moments: http://stackoverflow.com/questions/39403725/python-integrated-first-order-moment-of-2d-distribution
    y = np.argmax((X * np.abs(data)).sum(axis=1) / total)
    x = np.argmax((Y * np.abs(data)).sum(axis=0) / total)
    col = data[:, int(x)]
    sigma_y = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(y), :]
    sigma_x = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    offset = np.median(data.ravel())
    amplitude = data.max() - offset
    theta = 0
    params = [amplitude, x, y, sigma_x, sigma_y, theta, offset]
    return params

def fitgaussian(data):
    params = moments(data)
    errorfunction = lambda p: np.ravel(twodgaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


# Only fit for amplitude, x0, y0, sigma
def gaussian2D(amplitude, x0, y0, sigma):
    sigma2 = sigma*sigma
    return lambda x, y: amplitude * np.exp(-((x - x0) ** 2)/(2*sigma2)-((y - y0) ** 2)/(2*sigma2))

def init_params(data):
    total = np.abs(data).sum()
    Y, X = np.indices(data.shape)
    # first moments: http://stackoverflow.com/questions/39403725/python-integrated-first-order-moment-of-2d-distribution
    y = np.argmax((X * np.abs(data)).sum(axis=1) / total)
    x = np.argmax((Y * np.abs(data)).sum(axis=0) / total)
    col = data[:, int(x)]
    sigma_y = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(y), :]
    sigma_x = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    sigma = (sigma_x+sigma_y)/2
    offset = np.median(data.ravel())
    amplitude = data.max() - offset
    params = [amplitude, x, y, sigma]
    return params

def fit_gaussian2D(data):
    params = init_params(data)
    errorfunction = lambda p: np.ravel(gaussian2D(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(errorfunction, params)
    return p
