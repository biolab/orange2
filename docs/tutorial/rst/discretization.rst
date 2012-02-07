.. index:: discretization
.. index::
   single: discretization; entropy-based
.. index::
   single: discretization; frequency-based

Discretization
==============

Data discretization is a procedure that takes a data set and converts
all continuous attributes to categorical. In other words, it
discretizes the continuous attributes. Orange's core supports three
discretization methods: first using equal-width intervals
(``orange.EquiDistDiscretization``), second using equal-frequency
intervals (``orange.EquiNDiscretization``) and class-aware
discretization as introduced by Fayyad & Irani (AAAI92) that uses MDL
and entropy to find the best cut-off points
(``orange.EntropyDiscretization``). The discretization methods are
invoked through calling a preprocessor directive
``orange.Preprocessor_discretize`` which takes a data set and
discretization method, and returns a data set with any continuous
attribute being discretized.

In machine learning and data mining discretization may be used for
different purposes. It may be interesting to find informative cut-off
points in the data (for instance, finding that the cut-off for blood's
acidity is 7.3 may mean something to physicians).  In machine
learning, discretization may enable the use of some learning
algorithms (for instance, the naive Bayes in orange does not handle
continuous-valued attributes).

Data set discretization
-----------------------

Here is a script which demonstraters the basics of discretization in
Orange (:download:`disc.py <code/disc.py>`)::

   import orange
   
   def show_values(data, heading):
       print heading
       for a in data.domain.attributes:
           print "%s: %s" % (a.name, \
             reduce(lambda x,y: x+', '+y, [i for i in a.values]))
           
   data = orange.ExampleTable("iris")
   
   data_ent = orange.Preprocessor_discretize(data,\
     method=orange.EntropyDiscretization())
   show_values(data_ent, "Entropy based discretization")
   print
   
   data_n = orange.Preprocessor_discretize(data, \
     method=orange.EquiNDiscretization(numberOfIntervals=3))
   show_values(data_n, "Equal-frequency intervals")

Two types of discretization are used in the script, Fayyad-Irani's
method and equal-frequency interval discretization. The output of the
script is given bellow. Note that orange also changes the name of the
original attribute being discretized by adding *D_* at its
start. Further notice that with Fayyad-Irani's discretization, all
four attributes were found to have at least two meaningful cut-off
points::

   Entropy based discretization
   D_sepal length: <=5.50, (5.50000, 5.70000], >5.70
   D_sepal width: <=2.90, (2.90000, 3.00000], (3.00000, 3.30000], >3.30
   D_petal length: <=1.90, (1.90000, 4.70000], >4.70
   D_petal width: <=0.60, (0.60000, 1.70000], >1.70
   
   Equal-frequency intervals
   D_sepal length: <=5.35, (5.35000, 6.25000], >6.25
   D_sepal width: <=2.85, (2.85000, 3.15000], >3.15
   D_petal length: <=1.80, (1.80000, 4.85000], >4.85
   D_petal width: <=0.55, (0.55000, 1.55000], >1.55

Attribute-specific discretization
---------------------------------

In the example above, all continuous attributes were discretized using
the same method. This may be ok [in fact, this is how most often
machine learning people do discretization], but it may not be the
right way to do especially if you want to tailor discretization to
specific attributes. For this, you may want to apply different kind of
discretizations. The idea is that you discretize each of attributes
separately, and them use newly crafter attributes to form your new
domain for the new data set. We have not told you anything on working
with example domains, but trust us in what we are doing, and just read on.

In Orange, when converting examples (transforming one data set to
another), attribute's values can be computed from values of other
attributes, when needed. This is exactly how discretization
works. Let's take again the iris data set. We shall replace ``petal
width`` by quartile-discretized attribute called ``pl``. For ``sepal
length``, we'll keep the original attribute, but add the attribute
discretized using quartiles (``sl``) and using Fayyad-Irani's
algorithm (``sl_ent``). We shall also keep the original (continuous)
attribute ``sepal width`` (from :download:`disc2.py <code/disc2.py>`)::

   def printexamples(data, inxs, msg="First %i examples"):
     print msg % len(inxs)
     for i in inxs:
       print i, data[i]
     print
   
   import orange
   iris = orange.ExampleTable("iris")
   
   equiN = orange.EquiNDiscretization(numberOfIntervals=4)
   entropy = orange.EntropyDiscretization()
   
   pl = equiN("petal length", iris)
   sl = equiN("sepal length", iris)
   sl_ent = entropy("sepal length", iris)
   
   inxs = [0, 15, 35, 50, 98]
   d_iris = iris.select(["sepal width", pl, "sepal length",sl, sl_ent, iris.domain.classVar])
   printexamples(iris, inxs, "%i examples before discretization")
   printexamples(d_iris, inxs, "%i examples before discretization")

The output of this script is::

   5 examples before discretization
   0 [5.100000, 3.500000, 1.400000, 0.200000, 'Iris-setosa']
   15 [5.700000, 4.400000, 1.500000, 0.400000, 'Iris-setosa']
   35 [5.000000, 3.200000, 1.200000, 0.200000, 'Iris-setosa']
   50 [7.000000, 3.200000, 4.700000, 1.400000, 'Iris-versicolor']
   98 [5.100000, 2.500000, 3.000000, 1.100000, 'Iris-versicolor']
   
   5 examples before discretization
   0 [3.500000, '<=1.55', 5.100000, '(5.05, 5.75]', '<=5.50', 'Iris-setosa']
   15 [4.400000, '<=1.55', 5.700000, '(5.05, 5.75]', '(5.50, 6.10]', 'Iris-setosa']
   35 [3.200000, '<=1.55', 5.000000, '<=5.05', '<=5.50', 'Iris-setosa']
   50 [3.200000, '(4.45, 5.25]', 7.000000, '>6.35', '>6.10', 'Iris-versicolor']
   98 [2.500000, '(1.55, 4.45]', 5.100000, '(5.05, 5.75]', '<=5.50', 'Iris-versicolor']

Again, ``EquiNDiscretization`` and ``EntropyDiscretization`` are two
of the classes that perform different kinds of discretization, the
first will prepare four quartiles and the second does a Fayyad-Irani's
discretization based on entropy and MDL. Both are derived from a
common ancestor ``Discretization``; another discretization we could
use is ``EquiDistDiscretization`` that discretizes onto the given
number of intervals of equal width.

Called by an attribute (name, index or descriptor) and an example set,
discretization prepares a descriptor of a discretized attribute. The
constructed attribute is able to compute its value from value of the
original continuous attribute and this is why conversion by select can
work.

Names of discretized attribute's values tell the boundaries of the
interval. The output is thus informative, but not easily readable. You
can, however, always change names of values, as long as the number of
values remains the same. Adding the line::

   pl.values = sl.values = ["very low", "low", "high", "very high"]

to our code after the introduction of this two attributes (the new script is in
:download:`disc3.py <code/disc3.py>`), following is the second part of the output::

   5 examples before discretization
   0 [3.500000, 'very low', 5.100000, 'low', '<=5.50', 'Iris-setosa']
   15 [4.400000, 'very low', 5.700000, 'low', '(5.50, 6.10]', 'Iris-setosa']
   35 [3.200000, 'very low', 5.000000, 'very low', '<=5.50', 'Iris-setosa']
   50 [3.200000, 'high', 7.000000, 'very high', '>6.10', 'Iris-versicolor']
   98 [2.500000, 'low', 5.100000, 'low', '<=5.50', 'Iris-versicolor']

Want to know the cut-off points for the discretized attributes?  This
requires a little knowledge about the computation mechanics. How does
a discretized attribute know from each attribute it should compute its
values, and how? An attribute descriptor has a property
``getValueFrom`` which is a kind of classifier (it can indeed be a
classifier!) that is given an original example and returns the value
for the attribute. When converting examples from one domain to
another, the ``getValueFrom`` is called for all attributes of the new
domain that do not occur in the original. Get value takes the value of
the original attribute and calls a property transformer to discretize
it.

Both, ``EquiNDiscretization`` and ``EntropyDiscretization`` construct
transformer objects of type ``IntervalDiscretizer``. It's cut-off
points are stored in a list points (:download:`disc4.py <code/disc4.py>`)::

   import orange
   iris = orange.ExampleTable("iris")
   
   equiN = orange.EquiNDiscretization(numberOfIntervals=4)
   entropy = orange.EntropyDiscretization()
   
   pl = equiN("petal length", iris)
   sl = equiN("sepal length", iris)
   sl_ent = entropy("sepal length", iris)
   
   for attribute in [pl, sl, sl_ent]:
     print "Cut-off points for", attribute.name, \
       "are", attribute.getValueFrom.transformer.points
   
Here's the output::

   Cut-off points for D_petal length are <1.54999995232, 4.44999980927, 5.25>
   Cut-off points for D_sepal length are <5.05000019073, 5.75, 6.34999990463>
   Cut-off points for D_sepal length are <5.5, 6.09999990463>

Sometimes, you may not like the cut-offs suggested by functions in
Orange. In fact, we can tell that domain experts always like cut-offs
at least rounded, if not changed to completely something else. To do
this, simply assign new values to the cut-off points. Remember when
the new attribute is crafter (like ``sl``), this specifies only the
domain of the attribute and how it is derived. We did not created a
data set with this attribute yet, so before this, it is well time to
change anything the discretization will actually do to the data. In
the following example, we have rounded the cut-off points for the
attribute ``pl`` (:download:`disc5.py <code/disc5.py>`)::

   import orange
   iris = orange.ExampleTable("iris")
   
   equiN = orange.EquiNDiscretization(numberOfIntervals=4)
   entropy = orange.EntropyDiscretization()
   
   pl = equiN("petal length", iris)
   sl = equiN("sepal length", iris)
   sl_ent = entropy("sepal length", iris)
   
   points = pl.getValueFrom.transformer.points
   points2 = map(lambda x:round(x), points)
   pl.getValueFrom.transformer.points = points2
   
   for attribute in [pl, sl, sl_ent]:
     print "Cut-off points for", attribute.name, \
       "are", attribute.getValueFrom.transformer.points

.. note::
   ``pl`` is python's variable that stores the pointer to our
   attribute. The name of this attribute is derived from the name of
   original attribute (``petal length ``) by adding a prefix
   ``D_``. You may not like this, and you can change the name by
   assign its name to something else, like ``pl.name="pl"``.

.. warning::
   Don't try this with discretization when using
   ``EquiDistDiscretization``. Instead of ``IntervalDiscretizer`` this
   uses ``EquiDistDiscretizer`` with fields ``firstVal``, ``step`` and
   ``numberOfIntervals``.

Manual discretization
---------------------

What we have done above is something very close to manual
discretization, except that the number of intervals used was the same
as suggested by ``EquiNDiscretization``. To do everything manually, we
need to construct the same structures as the described discretization
algorithms. We need to define a descriptor, among with the ``name``,
``type``, ``values`` and ``getValueFrom``. The ``getValueFrom`` should
be ``IntervalDiscretizer`` and with it we specify the cut-off points.

Let's now discretize Iris' attribute pl using three intervals with
cut-off points 2.0 and 4.0 (:download:`disc6.py <code/disc6.py>`)::

   import orange
   
   def printexamples(data, inxs, msg="First %i examples"):
     print msg % len(inxs)
     for i in inxs:
       print data[i]
     print
   
   iris = orange.ExampleTable("iris")
   pl = orange.EnumVariable("pl")
   
   getValue = orange.ClassifierFromVar()
   getValue.whichVar = iris.domain["petal length"]
   getValue.classVar = pl
   getValue.transformer = orange.IntervalDiscretizer()
   getValue.transformer.points = [2.0, 4.0]
   
   pl.getValueFrom = getValue
   pl.values = ['low', 'medium', 'high']
   d_iris = iris.select(["petal length", pl, iris.domain.classVar])
   printexamples(d_iris, [0, 15, 35, 50, 98], "%i examples after discretization")
   
Notice that we have also named each of the three intervals, and
constructed the data set that shows both original and discretized
attribute::

   5 examples after discretization
   [1.400000, 'low', 'Iris-setosa']
   [1.500000, 'low', 'Iris-setosa']
   [1.200000, 'low', 'Iris-setosa']
   [4.700000, 'high', 'Iris-versicolor']
   [3.000000, 'medium', 'Iris-versicolor']

Applying discretization on the test set
---------------------------------------

In machine learning, you would often discretize the learning set. How
does one then apply the same discretization on the test set?  For
discretized attributes Orange remembers the how they were converted
from their original continuous versions, so you need only to convert
the testing examples to a new (discretized) domain. Following code
shows how (:download:`disc7.py <code/disc7.py>`)::

   import orange
   data = orange.ExampleTable("iris")
   
   #split the data to learn and test set
   ind = orange.MakeRandomIndices2(data, p0=6)
   learn = data.select(ind, 0)
   test = data.select(ind, 1)
   
   # discretize learning set, then use its new domain
   # to discretize the test set
   learnD = orange.Preprocessor_discretize(data, method=orange.EntropyDiscretization())
   testD = orange.ExampleTable(learnD.domain, test)
   
   print "Test set, original:"
   for i in range(3):
       print test[i]
   
   print "Test set, discretized:"
   for i in range(3):
       print testD[i]

Following is the output of the above script::

   Test set, original:
   [5.1, 3.5, 1.4, 0.2, 'Iris-setosa']
   [4.9, 3.0, 1.4, 0.2, 'Iris-setosa']
   [4.7, 3.2, 1.3, 0.2, 'Iris-setosa']
   Test set, discretized:
   ['<=5.50', '>3.30', '<=1.90', '<=0.60', 'Iris-setosa']
   ['<=5.50', '(2.90, 3.30]', '<=1.90', '<=0.60', 'Iris-setosa']
   ['<=5.50', '(2.90, 3.30]', '<=1.90', '<=0.60', 'Iris-setosa']



