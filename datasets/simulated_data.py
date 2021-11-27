import numpy as np

def generate_data(num=3, A=1.0, l=30, s=128, sigma=7):
    #l = 30
    p0 = np.array([0, l])
    angles = np.linspace(0, 360, num+1)[0:-1]
    pts = np.array([rot_pts(p0, angles[i]) for i in np.arange(num)])
    # Generate Gaussian data
    #s = 128
    #sigma = 7
    data = np.zeros((s,s))
    Y, X = np.ogrid[-s // 2:s // 2:1j * s, -s // 2:s // 2:1j * s]
    for (x, y) in pts:
        data = data + np.exp((-(X-x)**2-(Y-y)**2)/(2*sigma*sigma))
    data = data + A*np.exp((-(X)**2-(Y)**2)/(2*sigma*sigma))
    return data