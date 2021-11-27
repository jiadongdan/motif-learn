Dimension Reductions
====================

Dimension reduction is the transformation of the input data :math:`X \in \mathbb{R}^{n \times m}` into :math:`Y \in \mathbb{R}^{n \times k}` from :math:`m`-dimensional space to :math:`k`-dimensional space (:math:`k < m`). Related methods are commonly divided into linear and nonlinear approaches. The key difference between two genres is that if there exists a matrix :math:`P \in \mathbb{R}^{m \times k}` that statifies :math:`Y = XP`. In linear methods, the optimization is achived by obtaining :math:`P`, while in nonlinear methods, :math:`Y` is updated during the optimization.


Linear dimension reductions
---------------------------

.. admonition:: Definition: (Linear Dimensionality Reduction)

   Given :math:`n` :math:`m`-dimensional data points :math:`X = {x_1, x_2, \cdots, x_n} \in \mathbb{R}^{n \times m}` and a choice of dimensionality :math:`k < m`, optimize some objective :math:`f_{X}(\cdot)` to produce a linear transformation :math:`P \in \mathbb{R}^{m \times k}`, and call :math:`Y=P X \in \mathbb{R}^{r \times n}` the low dimensional transformed data.

The intuition behind the optimization program should be apparent: the objective :math:`f_{X}(\cdot)` defines the feature of interest to be captured in the data :math:`X`, and encodes some aspects of the linear mapping :math:`P` such that :math:`Y = P X`. Different objective functions will produce various :math:`P`. 

The most poopular linear dimensionality reduction technique is principal component analysis (PCA). The objective of PCA is finding a set of :math:`k` orthogonal basis that map X to Y, by maximizing the variance of Y in low-dimensional space. This is equivalent to minimizing the least-squares reconstruction error (the lost variance).


Nonlinear dimension reductions
------------------------------