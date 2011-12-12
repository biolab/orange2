.. index:: Pricipal Component Analysis
   
.. index:: 
   single: projection, Principal Component Analysis

*************************************
Pricipal Component Analysis (``pca``)
*************************************

An implementation of `principal component analysis <http://en.wikipedia.org/wiki/Principal_component_analysis>`_.
PCA uses an orthogonal transformation to transform input features into a set of uncorrelated features called principal
components. This transformation is defined in such a way that the first principal component has as high variance as
possible and each succeeding component in turn has the highest variance possible under constraint that is be orthogonal
to the preceding components.

Because PCA is sensitive to the relative scaling of the original variables the default behaviour of PCA class is to
standardize the input data.

Learner and Classifier
======================

.. index:: PCA
.. autoclass:: Orange.projection.pca.Pca
   :members:
   
.. autoclass:: Orange.projection.pca.PcaClassifier
   :members:

Examples
========

The following example demonstrates a straightforward invocation of PCA
(:download:`pca-run.py <code/pca-run.py>`, uses :download:`iris.tab <code/iris.tab>`):

.. literalinclude:: code/pca-run.py
   :lines: 7-

The call to the Pca constructor returns an instance of PcaClassifier, which is later used to transform data to PCA
feature space. Printing the classifier displays how much variance is covered with the first few components. Classifier
can also be used to access transformation vectors (eigen_vectors) and variance of the pca components (eigen_values).
Scree plot can be used when deciding, how many components to keep (:download:`pca-scree.py <code/pca-scree.py>`,
uses :download:`iris.tab <code/iris.tab>`):

.. literalinclude:: code/pca-scree.py
   :lines: 7-

.. image:: code/pca-scree.png
   :scale: 50 %
