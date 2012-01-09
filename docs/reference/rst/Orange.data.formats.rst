.. py:currentmodule:: Orange.data

=======================
Loading and saving data
=======================

Tab-delimited format
====================
Orange prefers to open data files in its native, tab-delimited format. This format allows us to specify type of features
and optional flags along with the feature names, which can ofter result in shorter loading times. This additional data
is provided in a form of a 3-line header. First line contains feature names, followed by type of features and optional
flags in that order.

Example of iris dataset in tab-delimited format (:download:`iris.tab <code/iris.tab>`)

.. literalinclude:: code/iris.tab
   :lines: 1-7

Feature types
-------------
 * discrete (or d) - imported as Orange.data.variable.Discrete
 * continuous (or c) - imported as Orange.data.variable.Continuous
 * string - imported as Orange.data.variable.String
 * basket - used for storing sparse data. More on basket formats in a dedicated section.

Optional flags
--------------
 * ignore (or i) - feature will not be imported
 * class (or c) - feature will be imported as class variable. Only one feature can be marked as class.
 * multiclass - feature is one of multiple classes. Data can have both, multiple classes and an ordinary class.
 * meta (or m) - feature will be imported as a meta attribute.
 * -dc


Other supported data formats
============================
Orange can import data from csv or tab delimited files where the first line contains attribute names followed by
lines containing data. For such files, orange tries to guess the type of features and treats the right-most
column as the class variable. If feature types are known in advance, special orange tab format should be used.
