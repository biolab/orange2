.. _PCA:

Principal Component Analysis
============================

.. image:: ../../../../Orange/OrangeWidgets/icons/Unknown.png
    :alt: PCA

Channels
--------

Inputs:
    - Input Data (Table)

Outputs:
    - Transformed Data (Table)
        PCA transformed input data.
    - Eigen Vectors (Table)
        Eigen vectors.


Description
-----------

`Principal Component Analysis`_ (PCA) computes the PCA linear transformation
of the input data.

.. _`Principal Component Analysis`: http://en.wikipedia.org/wiki/Principal_component_analysis

.. image:: images/PCA.png
    :alt: PCA widget


.. rst-class:: stamp-list

	1. Specifies the maximum number of principal components to select
	   (a special value 'All' specifies all the components)
	2. Specifies the number of components by the proportion of explained
	   variance  
	3. Output the transformed data and eigen vectors on any change to
	   the widget settings.
	4. Send the transformed data and eigen vectors.
	5. Graph of the explained variance by PCA componenets. The vertical
	   line can be dragged using the mouse.

The number of components of the transformation can be selected using either
the `Components Selection` input box or by dragging the vertical cutoff line
in the graph (|5|).
