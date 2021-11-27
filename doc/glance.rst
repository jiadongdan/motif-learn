Stempy at a Glance
===================

.. currentmodule:: stempy

Welcome to Stempy!

Stempy is a structure analysis toolbox for atomic resolution microscopy data. The strengths of Stempy are:

* Python-based -- Stempy is developed in Python, allowing for inspection and customization of all code in python and understandable python messages at run time
* NumPy based syntax for working with Zernike moments, points, nodes. 
* Matplotlib based visualization utilities -- Stempy enables interative data visualization, facilitates scientific figure generation, and aid acitve learning.

Code Breakdown
--------------

Let’s start the program. Here are the typical imports for a Stempy program.

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from stempy import io
>>> from stempy.plot import *
>>> from stempy.feature import *
>>> from stempy.manifold import *

Load Data
---------

>>> file_name = ''
>>> img= io.load_image(file_name)
>>> plt.imshow(img)

Find Feature Points
-------------------


Extract Local Patches
---------------------

Extract Zernike Features
------------------------

Embed Zernike Feature into 2D Space
-----------------------------------

Human-aided Labeling in Interative Mode
---------------------------------------

Automatic Clustering
--------------------



