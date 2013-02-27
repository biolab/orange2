.. _Interaction Graph:

Interaction Graph
=================

.. image:: ../icons/MDS.png

Graph of interactions between attributes

Signals
-------

Inputs:
   - Examples (ExampleTable)
      A table of examples

Outputs:
   - Selected Examples
      ???
   - Attribute Pair
      ???
   - Selected Attributes
      ???


The widget computes interactions between attributes as defined by Aleks Jakulin
in his work on `attribute interactions <http://stat.columbia.edu/~jakulin/Int/>`_.
The interaction is defined as the difference between the sum of individual
attribute information gains and the information gain of their cartesian
product. The interaction can be negative (e.g. when the attributes are
correlated), or positive (e.g. when the class is related to the xor of the
two attributes).

The widget uses an external application for drawing graphs,
`GraphViz <http://www.graphviz.org/>`_. It does not come with Orange, so you
will need to install separately for the entire widget to work.

The widget will be completely redesigned in the nearest future, so we here
only give its most basic description.


.. image:: images/InteractionGraph-Small.png

The widget is comprised of three parts. In the leftmost the user can select
the attributes among which she or he wants to compute the interactions. The
middle part contains all pairs of attributes - or all interesting pairs,
if :obj:`Show only important interactions` is checked. For attributes which
are in positive interaction, the blue parts at the left and the right end of
the bar represent the individual information gains and the green part in the
middle represents the interaction. For those in negative interaction, the red
part is the interaction, which can be interpreted as the amount of information
conveyed by both attributes, while the blue parts to the left and right are
each attribute's individual contribution.

The right part of the widget shows a graph of interactions.
