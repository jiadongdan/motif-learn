.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_moments.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_moments.py:


Plotting Zernike Moments
===================================

this is only testing



.. image:: /auto_examples/images/sphx_glr_plot_moments_001.png
    :alt: plot moments
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    D:\Dropbox\stempy\stempy\examples\plot_moments.py:102: UserWarning: In Matplotlib 3.3 individual lines on a stem plot will be added as a LineCollection instead of individual lines. This significantly improves the performance of a stem plot. To remove this warning and switch to the new behaviour, set the "use_line_collection" keyword argument to True.
      markerline, stemlines, baseline = ax.stem(x, y)






|


.. code-block:: default

    import numpy as np
    import matplotlib.pyplot as plt
    from stempy.feature import ZPs
    from scipy.ndimage import gaussian_filter

    def generate_h_axes(fig, n=2, y=0.1, left=1 / 15, right=1 / 30, wspace=0.25, ratios=None):
        if ratios == None:
            ratios = np.ones(n)
        elif np.isscalar(ratios):
            ratios = np.array([ratios]*n)
        elif np.iterable(ratios):
            ratios = np.array(ratios)
        width, height = fig.get_size_inches()
        ratio = width / height

        N = ratios.sum() + ratios.sum() / n * wspace * (n - 1)
        dx = (1 - left - right) / N
        ws = dx * ratios.sum() / n * wspace
        W = np.array([e * dx for e in ratios])
        h = W.min()
        axes = [fig.add_axes([left + W[0:i].sum() + i * ws, y, W[i], h * ratio]) for i in range(n)]
        return axes


    def plot_image(ax, img, clip=False, **kwargs):
        h, w = img.shape[0:2]
        im = ax.imshow(img, **kwargs)
        ax.axis('off')
        if clip == True:
            patch = plt.Circle((w / 2 - 0.5, h / 2 - 0.5), radius=(h + w) / 4 - 2, transform=ax.transData)
            im.set_clip_path(patch)

    def rotation_matrix(angle):
        angle = np.radians(angle)
        s = np.sin(angle)
        c = np.cos(angle)
        R = np.array([(c, -s), (s, c)])
        return process_zeros(R)


    def process_zeros(data, eps=1e-9):
        data[np.abs(data) < eps] = 0.0
        return data


    def rotate(pts, angle):
        pts = np.array(pts)
        R = rotation_matrix(angle)
        return np.dot(pts, R)

    def generate_synthetic_data():
        s = 128
        a = np.zeros(shape=(s, s))
    
        p = (0, s/3.5)
        sigma = s/20
        pts = np.array([rotate(p, i) for i in np.arange(0, 360, 120)]+[(0, 0)])
        pts[:, 0] += s//2
        pts[:, 1] += s//2
        for (x, y) in pts.astype(np.int):
            a[y, x] = 1
        a = gaussian_filter(a, sigma)
        return a/a.max()*255

    def shift_ax(ax, left=0, right=0, upper=0, lower=0):
        bbox = ax.get_position()
        x0, y0, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
        x = x0 + right - left
        y = y0 + upper - lower
        ax.set_position([x, y, w, h])

    def plot_decomposiiton(axes, ps, ss):
        zz = np.array(ps[1:])
        v_min = zz.min()
        v_max = zz.max()
        for i, ax in enumerate(axes):
            ax.axis('off')
            if i == 0:
                plot_image(ax, ps[i])
            if i == 1:
                shift_ax(ax, left=0.02)
            if i == 2:
                shift_ax(ax, left=0.03)
            if i == 3:
                shift_ax(ax, left=0.05)
            if i != 0:
                plot_image(ax, ps[i], clip=True, vmin=v_min, vmax=v_max)
                ax.text(-0.1, 0.5, ss[i], transform=ax.transAxes, va='center', ha='right', fontsize=8)

    def plot_moments(ax, y, color=None):
        if color is None:
            color = 'C0'
        x = np.arange(0, len(y))
        y = y/(np.abs(y).max())
        markerline, stemlines, baseline = ax.stem(x, y)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0.0, color='#0a0a0a', linestyle='-', alpha=0.7, lw=0.5)
        ax.tick_params(which="major", labelsize=10, direction='in', length=3, pad=1)
    
        markerline.set_mfc(color)
        markerline.set_mec(color)
        markerline.set_ms(2)
        for line in stemlines:
            line.set_color(color)
        baseline.set_visible(False)
        ax.set_ylabel('Normalized  '+r'$A_{n}^{m}$'+' [a.u.]')
        ax.set_xlabel(r'$j$')

    # generate synthetic image
    img = generate_synthetic_data()

    zps = ZPs(n_max=10, size=img.shape[0])
    y = zps.fit_transform(img)

    ss = ['$A_{0}^{0}$', r'$\approx A_{0}^{0}$', r'$+A_{1}^{-1}$', r'$+A_{1}^{1}$', r'$+\cdot\cdot\cdot+A_{10}^{10}$']

    fig = plt.figure(figsize=(7.2, 7.2/2), facecolor='#fcfcfc')
    axes = generate_h_axes(fig, n=5, y=0.65, left=1/15, right=1/30, wspace=0.55, ratios=1)

    ps = [img]+[e for e in zps.data]

    plot_decomposiiton(axes, ps, ss)

    left = 0.15
    ax = fig.add_axes([left, 0.15, 1-left-1/30, (1-left-1/30)*0.5])
    plot_moments(ax, y)

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.468 seconds)


.. _sphx_glr_download_auto_examples_plot_moments.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_moments.py <plot_moments.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_moments.ipynb <plot_moments.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
