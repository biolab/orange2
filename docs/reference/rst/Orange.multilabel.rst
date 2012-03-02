###########################################
Multi-label classification (``multilabel``)
###########################################

`Multi-label classification <http://en.wikipedia
.org/wiki/Multi-label_classification>`_ is a machine learning prediction
problem in which multiple binary variables (i.e. labels) are being predicted.
Orange offers a limited number of methods for this task.

Multi-label data is represented as :ref:`multi-target data <multiple-classes>`
with discrete binary classes with values '0' and '1'. Multi-target data is
also supported by Orange's tab file format
using :ref:`multiclass directive <tab-delimited>`.

.. automodule:: Orange.multilabel.br
.. automodule:: Orange.multilabel.lp
.. automodule:: Orange.multilabel.multiknn
.. automodule:: Orange.multilabel.mlknn
.. automodule:: Orange.multilabel.brknn
