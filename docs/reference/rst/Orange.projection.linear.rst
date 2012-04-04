.. automodule:: Orange.projection.linear

##############################
Linear projection (``linear``)
##############################

.. index:: linear projection

.. index::
   single: projection; linear

Linear transformation of the data might provide a unique insight into the data through observation of the optimized
projection or through visualization of the space with reduced dimensionality.

This module contains the FreeViz linear projection optimization algorithm
[1], PCA and FDA and utility classes for classification of instances based on
kNN in the linearly transformed space.

Methods in this module use given data set to optimize a linear projection of features into a new vector space. The
transformation is returned as a :class:`~Orange.projection.linear.Projector` instance that, when invoked, projects
any given data with the domain that matches the domain that was used to optimize the projection.

.. autoclass:: Orange.projection.linear.Projector
   :members:

*************************************
Pricipal Component Analysis (``pca``)
*************************************

.. index:: Pricipal Component Analysis

.. index::
   single: projection, Principal Component Analysis

`PCA <http://en.wikipedia.org/wiki/Principal_component_analysis>`_ uses an orthogonal transformation to transform input
features into a set of uncorrelated features called principal
components. This transformation is defined in such a way that the first principal component has as high variance as
possible and each succeeding component in turn has the highest variance possible under constraint that it is orthogonal
to the preceding components.

Because PCA is sensitive to the relative scaling of the original variables, the default behaviour of PCA class is to
standardize the input data.

Optimizer and Projector
=======================

.. index:: PCA
.. autoclass:: Orange.projection.linear.PCA
   :members:

.. autoclass:: Orange.projection.linear.PcaProjector
   :members:
   :show-inheritance:

Examples
========

The following example demonstrates a straightforward invocation of PCA
(:download:`pca-run.py <code/pca-run.py>`):

.. literalinclude:: code/pca-run.py
   :lines: 7-

The call to the Pca constructor returns an instance of PcaClassifier, which is later used to transform data to PCA
feature space. Printing the classifier displays how much variance is covered with the first few components. Classifier
can also be used to access transformation vectors (eigen_vectors) and variance of the pca components (eigen_values).
Scree plot can be used when deciding, how many components to keep (:download:`pca-scree.py <code/pca-scree.py>`):

.. literalinclude:: code/pca-scree.py
   :lines: 7-

.. image:: files/pca-scree.png
   :scale: 50 %


.. index:: Fisher Discriminant Analysis

.. index::
   single: projection, Fisher Discriminant Analysis

**************************************
Fisher discriminant analysis (``fda``)
**************************************

As a variant of LDA (Linear Discriminant Analysis),
`FDA <http://en.wikipedia.org/wiki/Linear_discriminant_analysis#Fisher.27s_linear_discriminant>`_ finds
a linear combination of features
that separates two or more classes best.

Optimizer and Projector
=======================

.. index:: FDA
.. autoclass:: Orange.projection.linear.Fda
   :members:

.. autoclass:: Orange.projection.linear.FdaProjector
   :members:
   :show-inheritance:

*******
FreeViz
*******

Freeviz
`(Demsar et al, 2005) <http://www.ailab.si/idamap/idamap2005/papers/12%20Demsar%20CR.pdf>`_
is a method that
finds a good two-dimensional linear projection of the given data, where the
quality is defined by a separation of the data from different classes and the
proximity of the instances from the same class. FreeViz would normally be used
through a widget since it is primarily a method for graphical exploration of
the data. About the only case where one would like to use this module directly
is to tests the classification aspects of the method, that is, to verify the
accuracy of the resulting kNN-like classifiers on a set of benchmark data sets.

Description of the method itself is far beyond the scope of this page. See the
above paper for the original version of the method; at the moment of writing
the method has been largely extended and not published yet, though the basic
principles are the same.

[1] Janez Demsar, Gregor Leban, Blaz Zupan: FreeViz - An Intelligent
Visualization Approach for Class-Labeled Multidimensional Data Sets,
Proceedings of IDAMAP 2005, Edinburgh.

.. autoclass:: Orange.projection.linear.FreeViz
   :members:
   :show-inheritance:
   :exclude-members: attractG, attractG, autoSetParameters, cancelOptimization,
      classPermutationList, classPermutationList, findProjection,
      forceBalancing, forceSigma, getShownAttributeList, mirrorSymmetry,
      optimizeSeparation, optimize_FAST_Separation, optimize_LDA_Separation,
      optimize_SLOW_Separation, radialAnchors, randomAnchors, repelG,
      s2nMixAnchors, s2nMixData, s2nPlaceAttributes, s2nSpread,
      setStatusBarText, showAllAttributes, stepsBeforeUpdate,
      useGeneralizedEigenvectors

:class:`~Orange.projection.linear.FreeViz` can be used in code to optimize
a linear projection to two dimensions:

.. literalinclude:: code/freeviz-projector.py
   :lines: 7-

Learner and Classifier
======================

.. autoclass:: Orange.projection.linear.FreeVizLearner
   :members:
   :show-inheritance:

.. autoclass:: Orange.projection.linear.FreeVizClassifier
   :members:
   :show-inheritance:

.. autoclass:: Orange.projection.linear.S2NHeuristicLearner
   :members:
   :show-inheritance:
