How Forces Work
===============

The Anatomy of the Force-directed Emebedding Algorithm
-------------------------------------------------------

There are **three** important stages in our force-directed embedding method. This section is to help the readers have quick preview of the source code and reproduce the results with better performance.

* Compute the graph (*adjacent matrix*) from the input high dimensional data :math:`X`. 
* Generate ``nodes`` (vertices) from initial layout and ``pairs`` from the graph.
* Optimize the position of ``nodes`` (vertices) under specified forces.

.. topic:: Compute a graph from input feature vectors :math:`X`

   1. use *k-nearest neighbors* (kNN) model to obtain the neighbor distances and neigbor indices (locate the valid entries of graph adjacency matrix)
   2. use asssumptions in UMAP to calculate assymetric weight of the adjacency matrix :math:`P_{ij}`
   3. symmetrize the adjacency matrix

.. graphviz:: /graphviz/algo1.dot

.. topic:: Generate ``nodes`` and ``pairs`` from the graph

   For performance and clarity, the ``nodes`` are *structured array* (:py:class:`numpy.ndarray`). In this way, it is convenient to use literal string *'x'* and *'y'* to access the coordinates.

   >>> xy = [(0., 0.), (1., 1.)]
   >>> node_dtype = np.dtype([('x', np.float64), ('y', np.float64)])
   >>> # Convert to structured array
   >>> nodes = np.array(xy, dtype=node_dtype)
   >>> # access x and y
   >>> nodes['x']
   >>> nodes['y']

   

   *pairs* are numpy arrays of the shape ``(n, 3)``, where ``n`` is the number of nonzero values in adjacency matrix. Every row of pairs is contructed in the form ``(i, j, w)``, where ``i`` and ``j`` are indices of :math:`i^{th}` and :math:`j^{th}` nodes and ``w`` the weight of the two nodes.

.. graphviz:: /graphviz/algo2.dot

.. topic:: Layout optimization under forces

   This is where most of the computation happens.


   