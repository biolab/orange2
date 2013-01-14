.. _Hierarchical Clustering:

Hierarchical Clustering
=======================

.. image:: ../icons/HierarchicalClustering.png

Groups items using a hierarchical clustering algorithm.

Signals
-------

Inputs:
   - Distance Matrix
      A matrix of distances between items being clustered


Outputs:
   - Selected Examples
      A list of selected examples; applicable only when the input matrix refers to distances between examples
   - Structured Data Files
      ???


Description
-----------

The widget computes hierarchical clustering of arbitrary types of objects from the matrix of distances between them and shows the corresponding dendrogram. If the distances apply to examples, the widget offers some special functionality (adding cluster indices, outputting examples...).

.. image:: images/HierarchicalClustering.png

The widget supports three kinds of linkages. In :obj:`Single linkage` clustering, the distance between two clusters is defined as the distance between the closest elements of the two clusters. :obj:`Average linkage` clustering computes the average distance between elements of the two clusters, and :obj:`Complete linkage` defines the distance between two clusters as the distance between their most distant elements.

Nodes of the dendrogram can be labeled. What the labels are depends upon the items being clustered. For instance, when clustering attributes, the labels are obviously the attribute names. When clustering examples, we can use the values of one of the attributes, typically one that give the name or id of an instance, as labels. The label can be chosen in the box :obj:`Annotate`, which also allows setting the font size and line spacing.

Huge dendrograms can be pruned by checking :obj:`Limit pring depth` and selecting the appropriate depth. This only affects the displayed dendrogram and not the actual clustering.

Clicking inside the dendrogram can have two effects. If the cut off line is not shown (:obj:`Show cutoff line` is unchecked), clicking inside the dendrogram will select a cluster. Multiple clusters can be selected by holding Ctrl. Each selected cluster is shown in different color and is treated as a separate cluster on the output.

If :obj:`Show cutoff line` is checked, clicking in the dendrogram places a cutoff line. All items in the clustering are selected and the are divided into groups according to the position of the line.

If the items being clustered are examples, they can be added a cluster index (:obj:`Append cluster indices`). The index can appear as a :obj:`Class attribute`, ordinary :obj:`Attribute` or a :obj:`Meta attribute`. In the former case, if the data already has a class attribute, the original class is placed among meta attributes.

The data can be output on any change (:obj:`Commit on change`) or, if this is disabled, by pushing :obj:`Commit`.


Clustering has two parameters that can be set by the user, the number of clusters and the type of distance metrics, :obj:`Euclidean distance` or :obj:`Manhattan`. Any changes must be confirmed by pushing :obj:`Apply`.

The table on the right hand side shows the results of clustering. For each cluster it gives the number of examples, its fitness and BIC.

Fitness measures how well the cluster is defined. Let d<sub>i,C</sub> be the average distance between point i and the points in cluster C. Now, let a<sub>i</sub> equal d<sub>i,C'</sub>, where C' is the cluster i belongs to, and let b<sub>i</sub>=min d<sub>i,C</sub> over all other clusters C. Fitness is then defined as the average silhouette of the cluster C, that is avg( (b<sub>i</sub>-a<sub>i</sub>)/max(b<sub>i</sub>, a<sub>i</sub>) ).

To make it simple, fitness close to 1 signifies a well-defined cluster.

BIC is short for Bayesian Information Criteria and is computed as ln L-k(d+1)/2 ln n, where k is the number of clusters, d is dimension of data (the number of attributes) and n is the number of examples (data instances). L is the likelihood of the model, assuming the spherical Gaussian distributions around the centroid(s) of the cluster(s).


Examples
--------

The schema below computes clustering of attributes and of examples.

.. image:: images/HierarchicalClustering-Schema.png

We loaded the Zoo data set. The clustering of attributes is already shown above. Below is the clustering of examples, that is, of animals, and the nodes are annotated by the animals' names. We connected the `Linear projection widget <../Visualize/LinearProjection.htm>`_ showing the freeviz-optimized projection of the data so that it shows all examples read from the file, while the signal from Hierarchical clustering is used as a subset. Linear projection thus marks the examples selected in Hierarchical clustering. This way, we can observe the position of the selected cluster(s) in the projection.

.. image:: images/HierarchicalClustering-Example.png

To (visually) test how well the clustering corresponds to the actual classes in the data, we can tell the widget to show the class ("type") of the animal instead of its name (:obj:`Annotate`). Correspondence looks good.

.. image:: images/HierarchicalClustering-Example2.png

A fancy way to verify the correspondence between the clustering and the actual classes would be to compute the chi-square test between them. As Orange does not have a dedicated widget for that, we can compute the chi-square in `Attribute Distance <AttributeDistance.htm>`_ and observe it in `Distance Map <DistanceMap.htm>`_. The only caveat is that Attribute Distance computes distances between attributes and not the class and the attribute, so we have to use `Select attributes <../Data/SelectAttributes.htm>`_ to put the class among the ordinary attributes and replace it with another attribute, say "tail" (this is needed since Attribute Distance requires data with a class attribute, for technical reasons; the class attribute itself does not affect the computed chi-square).

A more direct approach is to leave the class attribute (the animal type) as it is, simply add the cluster index and observe its information gain in the `Rank widget <../Data/Rank.htm>`_.

More tricks with a similar purpose are described in the documentation for `K-Means Clustering <K-MeansClustering.htm>`_.

The schema that does both and the corresponding settings of the hiearchical clustering widget are shown below.

.. image:: images/HierarchicalClustering-Schema2.png

.. image:: images/HierarchicalClustering-Example3.png
