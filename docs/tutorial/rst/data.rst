The Data
========

.. index: data

This section describes how to load and save the data. We also show how to explore the data, its domain description, how to report on basic data set statistics, and how to sample the data.

Data Input
----------

.. index:: 
   single: data; input

Orange can read files in native and other data formats. Native format starts with feature (attribute) names, their type (continuous, discrete, string). The third line contains meta information to identify dependent features (class), irrelevant features (ignore) or meta features (meta). Here are the first few lines from a data set :download:`lenses.tab <code/lenses.tab>` on prescription of eye
lenses [CJ1987]::

   age       prescription  astigmatic    tear_rate     lenses
   discrete  discrete      discrete      discrete      discrete 
                                                       class
   young     myope         no            reduced       none
   young     myope         no            normal        soft
   young     myope         yes           reduced       none
   young     myope         yes           normal        hard
   young     hypermetrope  no            reduced       none


Values are tab-limited. The data set has four attributes (age of the patient, spectacle prescription, notion on astigmatism, and information on tear production rate) and an associated three-valued dependent variable encoding lens prescription for the patient (hard contact lenses, soft contact lenses, no lenses). Feature descriptions could use one letter only, so the header of this data set could also read::

   age       prescription  astigmatic    tear_rate     lenses
   d         d             d             d             d 
                                                       c

The rest of the table gives the data. Note that there are 5
instances in our table above (check the original file to see
other). Orange is rather free in what attribute value names it
uses, so they do not need all to start with a letter like in our
example.

You may download :download:`lenses.tab <code/lenses.tab>` to a target directory and there open a python shell. Alternatively, just execute the code below; this particular data set comes with Orange instalation, and Orange knows where to find it:

    >>> import Orange
    >>> data = Orange.data.Table("lenses")
    >>>

Note that for the file name no suffix is needed; as Orange checks if any files in the current directory are of the readable type. The call to ``Orange.data.Table`` creates an object called ``data`` that holds your data set and information about the lenses domain:

>>> print data.domain.features
<Orange.feature.Discrete 'age', Orange.feature.Discrete 'prescription', Orange.feature.Discrete 'astigmatic', Orange.feature.Discrete 'tear_rate'>
>>> print data.domain.class_var
Orange.feature.Discrete 'lenses'
>>> for d in data[:3]:
   ...:     print d
   ...:
['young', 'myope', 'no', 'reduced', 'none']
['young', 'myope', 'no', 'normal', 'soft']
['young', 'myope', 'yes', 'reduced', 'none']
>>>

The following script wraps-up everything we have done so far and lists first 5 data instances with ``soft`` perscription:

.. literalinclude:: code/data-lenses.py

Note that data is an object that holds both the data and information on the domain. We show above how to access attribute and class names, but there is much more information there, including that on feature type, set of values for categorical features, and other.

Saving the Data
---------------

Data objects can be saved to a file:

>>> data.save("new_data.tab")
>>>

This time, we have to provide the extension for Orange to know which data format to use. An extension for native Orange's data format is ".tab". The following code saves only the data items with myope perscription:

.. literalinclude:: code/data-save.py

Exploration of Data Domain
--------------------------

.. index::
   single: data; features
.. index::
   single: data; domain
.. index::
   single: data; class

Data table object stores information on data instances as well as on data domain. Domain holds the names of features, optional classes, their types and, if categorical, value names.

.. literalinclude:: code/data-domain1.py

Orange's objects often behave like Python lists and dictionaries, and can be indexed or accessed through feature names.

.. literalinclude:: code/data-domain2.py
    :lines: 5-

Data Instances
--------------

.. index::
   single: data; instances
.. index::
   single: data; examples

Data table stores data instances (or examples). These can be index or traversed as any Python list. Data instances can be considered as vectors, accessed through element index, or through feature name.

.. literalinclude:: code/data-instances1.py

The script above displays the following output::

   First three data instances:
   [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
   [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']
   [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']
   25-th data instance:
   [4.8, 3.4, 1.9, 0.2, 'Iris-setosa']
   Value of 'sepal width' for the first instance: 3.5
   The 3rd value of the 25th data instance: 1.9

Iris data set we have used above has four continous attributes. Here's a script that computes their mean:

.. literalinclude:: code/data-instances2.py
   :lines: 3-

Above also illustrates indexing of data instances with objects that store features; in ``d[x]`` variable ``x`` is an Orange object. Here's the output::

   Feature         Mean
   sepal length    5.84
   sepal width     3.05
   petal length    3.76
   petal width     1.20


Slightly more complicated, but more interesting is a code that computes per-class averages:

.. literalinclude:: code/data-instances3.py
   :lines: 3-

Of the four features, petal width and length look quite discriminative for the type of iris::

   Feature             Iris-setosa Iris-versicolor  Iris-virginica
   sepal length               5.01            5.94            6.59
   sepal width                3.42            2.77            2.97
   petal length               1.46            4.26            5.55
   petal width                0.24            1.33            2.03

Finally, here is a quick code that computes the class distribution for another data set:

.. literalinclude:: code/data-instances4.py

Missing Values
--------------

.. index::
   single: data; missing values

Consider the following exploration of senate voting data set::

   >>> data = Orange.data.Table("voting.tab")
   >>> data[2]
   ['?', 'y', 'y', '?', 'y', 'y', 'n', 'n', 'n', 'n', 'y', 'n', 'y', 'y', 'n', 'n', 'democrat']
   >>> data[2][0].is_special()
   1
   >>> data[2][1].is_special()
   0

The particular data instance included missing data (represented with '?') for first and fourth feature. We can use the method ``is_special()`` to detect parts of the data which is missing. In the original data set file, the missing values are, by default, represented with a blank space. We use the method ``is_special()`` below to examine each feature and report on proportion of instances for which this feature was undefined:

.. literalinclude:: code/data-missing.py

First few lines of the output of this script are::

    2.8% handicapped-infants
   11.0% water-project-cost-sharing
    2.5% adoption-of-the-budget-resolution
    2.5% physician-fee-freeze
    3.4% el-salvador-aid

A single-liner that reports on number of data instances with at least one missing value is::

    >>> sum(any(d[x].is_special() for x in data.domain.features) for d in data)
    203


Data Subsetting
---------------

.. index::
   single: data; subsetting

``Orange.data.Table`` accepts a list of data items and returns a new data set. This is useful for any data subsetting:

.. literalinclude:: code/data-subsetting.py
   :lines: 3-

The code outputs::

   Subsetting from 150 to 99 instances.

and inherits the data description (domain) from the original data set. Changing the domain requires setting up a new domain descriptor. This feature is useful for any kind of feature selection:

.. literalinclude:: code/data-featureselection.py
   :lines: 3-

.. index::
   single: feature; selection

By default, ``Orange.data.Domain`` assumes that last feature in argument feature list is a class variable. This can be changed with an optional argument::

   >>> nd = Orange.data.Domain(data.domain.features[:2], False)
   >>> print nd.class_var
   None
   >>> nd = Orange.data.Domain(data.domain.features[:2], True)
   >>> print nd.class_var
   Orange.feature.Continuous 'sepal width'

The first call to ``Orange.data.Domain`` constructed the classless domain, while the second used the last feature and constructed the domain with one input feature and a continous class.   

**References**

.. [CJ1987] Cendrowska J (1987) PRISM: An algorithm for inducing modular rules, International Journal of Man-Machine Studies, 27, 349-370.
