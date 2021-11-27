import numbers
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

from .lattice import generate_lattice

from ..cuda.cuda_helper import calculate_block_dim
from ..cuda.cuda_helper import calculate_grid_dim

from ..io import load_pickle

import os

kernel = SourceModule("""
__global__ void make_blobs(float* matrix, float2 *pts, int num_pts, float sigma, int rows, int cols, float A) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	float xf = x + 0.5f;
	float yf = y + 0.5f;
	if (x < cols && y < rows) {
		size_t  idx = y*cols + x;
		float temp = 0;
		for (int i = 0; i < num_pts; i++) {
			float x_0 = pts[i].x;
			float y_0 = pts[i].y;
			temp += A*exp(-(pow(xf - x_0, 2) + pow(yf - y_0, 2)) / (2 * sigma*sigma));
		}
		matrix[idx] = temp;
	}
}
""")


def make_blobs(pts, sigma, A, img_shape):
    if isinstance(img_shape, numbers.Number):
        img_shape = tuple([img_shape] * 2)
    if pts.dtype != np.float32:
        pts = pts.astype(np.float32)
    pts_gpu = gpuarray.empty(pts.shape[0], dtype=gpuarray.vec.float2)
    cuda.memcpy_htod(pts_gpu.gpudata, pts.data)
    num_pts = np.array(pts.shape[0], dtype=np.int)
    sigma = np.array(sigma, dtype=np.float32)
    A = np.array(A, dtype=np.float32)
    rows, cols = np.array(img_shape, dtype=np.int)
    data_gpu = gpuarray.empty(rows*cols, dtype=np.float32)
    block_dim = calculate_block_dim(rows, cols)
    grid_dim = calculate_grid_dim(rows, cols, block_dim)
    func = kernel.get_function("make_blobs")
    func(data_gpu, pts_gpu, num_pts, sigma, rows, cols, A, block=block_dim, grid=grid_dim)
    return data_gpu.get().reshape(rows, cols)


def generate_sample():
    u1 = (45, 25.981)
    v1 = (45, -25.981)
    p1 = (0, 0)
    pts_A = generate_lattice((1024, 1024), p1, u1, v1)
    u2 = (45, 25.981)
    v2 = (45, -25.981)
    p2 = (30, 0)
    pts_B = generate_lattice((1024, 1024), p2, u2, v2)
    A = make_blobs(pts_A, 7.5, 1.0, 1024)
    B = make_blobs(pts_B, 7, 0.8, 1024)
    return A+B

def lattice():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/lattice.pkl'
    return load_pickle(file_name)

def periodic():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/periodic.pkl'
    return load_pickle(file_name)

def mose2():
    data_dir = os.path.dirname(__file__)
    file_name = data_dir + '/MoSe2.pkl'
    return load_pickle(file_name)

def add_gaussian_noise(img, sigma=0.2):
    mean = np.mean(img)
    noise = np.random.normal(0, sigma**2, img.shape)
    return img+noise

def add_poisson_noise(img, lmda = 10):
    return np.random.poisson(img * lmda)/lmda

def generate_gaussian_noise(sigma, img_shape, mean=0):
    noise = np.random.normal(0, sigma, img_shape)
    return noise
