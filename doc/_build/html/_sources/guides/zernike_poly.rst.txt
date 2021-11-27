Dimension Reduction via Zernike Polynomials
===========================================

.. currentmodule:: stempy.feature

What is Zernike Polynomials
---------------------------

The Zernike polynomials (ZP) are a complete set of **orthogonal** basis functions defined in the unit disk denoted by the double indexing scheme :math:`Z_{n}^{m}`, where :math:`n` is a nonnegative integer, and :math:`m=\{-n,-n+2,-n+4, \cdots ,n\}` for a given :math:`n`. The double indices :math:`(n, n)` are ordered into a single index  :math:`j=(n(n+2)+m)/2`. Each ZP consists of a normalization term :math:`N_{n}^{m}`, a radial term :math:`R_{n}^{|m|}` , and an azimuthal term :math:`\sin(m\theta)` or :math:`\cos(m\theta)`:

.. math::
   Z_{n}^{m}(\rho, \theta)=\left\{\begin{array}{ll}
   N_{n}^{m} R_{n}^{|m|}(\rho) \cos (m \theta) ; & \text { for } m \geq 0 \\
   -N_{n}^{m} R_{n}^{|m|}(\rho) \sin (m \theta) ; & \text { for } m<0
   \end{array}\right..


Here, :math:`R_{n}^{|m|}`  and :math:`N_{n}^{m}` are given by

.. math::
   R_{n}^{|m|}(\rho)=\sum_{k=0}^{(n-|m|) / 2} \frac{-1^{k}(n-k) !}{k !\left(\frac{m+|m|}{2}-k\right) !\left(\frac{n-|m|}{2}-k\right) !} \rho^{n-2 k}

and

.. math::
   N_{n}^{m}=\sqrt{\frac{2(n+1)}{1+\delta_{m 0}}},

where :math:`\delta_{m 0}` is the Kronecker delta.


Stempy provides the class :py:class:`ZPs` to generate a series of polynomials. By specifying the maximum radial index ``n_max``, and the size of polynomial array ``size``, the Zernike polynomials can be initialized and array data can be accessed by its attribute ``data``. Moreover, the user can select polynomials with different symmetry characteristics if the ``states`` parameter is given. For example, :code:`states = 3` will only select polynomial terms with :math:`|m|=3`, and :code:`states = [3, 6]` will select polynomial terms with :math:`|m|=3, 6`. 

.. code-block:: python
   :linenos:

   from stempy.feature import ZPs
   
   # intialize zps
   zps = ZPs(n_max=10, size=256, states=None)
   # polynomial arrays
   data = zps.data
   # n indices
   n = zps.n
   # m indices
   m = zps.m

Visualization of Zernike Polynomials
------------------------------------

.. image:: /auto_examples/images/sphx_glr_plot_zps_001.png
   :target: ../../html/auto_examples/plot_zps.html
   :align: center
   :scale: 80

The first 21 terms of Zernike polynomials arranged in a pyramid form. Each polynomial term is labelled by :math:`Z_n^m`, where `n` is the radial index and `m` is the azimuthal index. All Zernike polynomials are vertically arranged by `n` and horizontally ordered by `m`.

The array data of Zernike polynomials can be visualized above and they show prominent symmetry characteristics.


Zernike Moments
---------------

Any square-integrable functions :math:`f(\rho, \theta)` within a unit disk can be decomposed into an infinite series comprising weighted Zernike polynomials:

.. math::
   f(\rho, \theta)=\sum_{n=0}^{\infty} \sum_{m=-n}^{n} A_{n}^{m} Z_{n}^{m}(\rho, \theta), \quad n-|m|=\text { even }

where the coefficients :math:`A_{n}^{m}` is can be calculated as,

.. math::
   A_{n}^{m}=\int_{0}^{2 \pi} \int_{0}^{1} f(\rho, \theta) Z_{n}^{m}(\rho, \theta) \rho \mathrm{d} \rho \mathrm{d} \theta .



Visualization of Zernike Moments
--------------------------------

.. image:: /auto_examples/images/sphx_glr_plot_moments_001.png
   :target: ../../html/auto_examples/plot_moments.html
   :align: center
   :scale: 80

Zernike Representation as Matrix Approximation
----------------------------------------------



