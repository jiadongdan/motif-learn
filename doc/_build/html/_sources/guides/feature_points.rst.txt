Feature points
==============


Feature and Feature Vector

A feature is an individual measurable property or characteristic of a phenomenon being observed. A set of numeric features can be conveniently described by a feature vector.


Notations
---------

* :math:`\mathbf{x}_{i}`: 
* :math:`\mathbf{X}`: 
* :math:`\mathbf{Y}`:




Feature points in atomic resolution microscopy images are points with prominant symmetry characteristics. In general, they are positions with local maximum intensity.

In this package, feature points are represented by :py:class:`stempy.feature.ptsarray`, which is a subclass of :py:class:`numpy.ndarray`. 


Find Feature Points
-------------------

The aberration-corrected scanning transmission electron microscope (STEM) has great sensitivity to environmental or instrumental disturbances such as acoustic, mechanical, or electromagnetic interference. This interference can introduce distortions to the images recorded and degrade both signal noise and resolution performance. These types of noise make it difficult to detect robust positions from all available atomic columns. 

**Stempy** provide a robust peak finding pipeline which uses image as only input to retrieve the positions of all atomic columns. It is implemented by the function :py:func:`stempy.feature.local_max`, which returns the coordinates of local intensity maxima in an image. There are two major steps: 1. obtain a smoothed version of image using local threshold method, 2. retrive all positions of intensity maxima by a maximum filters. It is noted that :py:func:`stempy.feature.local_max` returns integer coordinates.

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from stempy import io
>>> 
>>> # load the image data
>>> file_name = ''
>>> img = io.load_image(file_name)
>>>
>>> # peak finding
>>> pts = local_max(img)
>>> 
>>> # display the results
>>> fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))
>>> ax.imshow(img)
>>> ax.plot(pts[:, 0], pts[:, 1], 'r.')

