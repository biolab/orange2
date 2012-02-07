Basic data exploration
======================

.. index:: basic data exploration

Until now we have looked only at data files that include solely
nominal (discrete) attributes. Let's make thinks now more interesting,
and look at another file with mixture of attribute types. We will
first use adult data set from UCI ML Repository. The prediction task
related to this data set is to determine whether a person
characterized by 14 attributes like education, race, occupation, etc.,
makes over $50K/year. Because of the original set :download:`adult.tab <code/adult.tab>` is
rather big (32561 data instances, about 4 MBytes), we will first
create a smaller sample of about 3% of instances and use it in our
examples. If you are curious how we do this, here is the code
(:download:`sample_adult.py <code/sample_adult.py>`)::

   import orange
   data = orange.ExampleTable("adult")
   selection = orange.MakeRandomIndices2(data, 0.03)
   sample = data.select(selection, 0)
   sample.save("adult_sample.tab")

Above loads the data, prepares a selection vector of length equal to
the number of data instances, which includes 0's and 1's, but it is
told that there should be about 3% of 0's. Then, those instances are
selected which have a corresponding 0 in selection vector, and stored
in an object called *sample*. The sampled data is then saved in a
file.  Note that ``MakeRandomIndices2`` performs a stratified selection,
i.e., the class distribution of original and sampled data should be
nearly the same.

Basic characteristics of data sets
----------------------------------

.. index::
   single: basic data exploration; attributes
.. index::
   single: basic data exploration; classes
.. index::
   single: basic data exploration; missing values

For classification data sets, basic data characteristics are most
often number of classes, number of attributes (and of these, how many
are nominal and continuous), information if data contains missing
values, and class distribution. Below is the script that does all
this (:download:`data_characteristics.py <code/data_characteristics.py>`, :download:`adult_sample.tab <code/adult_sample.tab>`)::

   import orange
   data = orange.ExampleTable("adult_sample")
   
   # report on number of classes and attributes
   print "Classes:", len(data.domain.classVar.values)
   print "Attributes:", len(data.domain.attributes), ",",
   
   # count number of continuous and discrete attributes
   ncont=0; ndisc=0
   for a in data.domain.attributes:
       if a.varType == orange.VarTypes.Discrete:
           ndisc = ndisc + 1
       else:
           ncont = ncont + 1
   print ncont, "continuous,", ndisc, "discrete"
   
   # obtain class distribution
   c = [0] * len(data.domain.classVar.values)
   for e in data:
       c[int(e.getclass())] += 1
   print "Instances: ", len(data), "total",
   for i in range(len(data.domain.classVar.values)):
       print ",", c[i], "with class", data.domain.classVar.values[i],
   print

The first part is the one that we know already: the script import
Orange library into Python, and loads the data. The information on
domain (class and attribute names, types, values, etc.) are stored in
``data.domain``. Information on class variable is accessible through the
``data.domain.classVar`` object which stores
a vector of class' values. Its length is obtained using a function
``len()``. Similarly, the list of attributes is stored in
data.domain.attributes. Notice that to obtain the information on i-th
attribute, this list can be indexed, e.g., ``data.domain.attributes[i]``.

To count the number of continuous and discrete attributes, we have
first initialized two counters (``ncont``, ``ndisc``), and then iterated
through the attributes (variable ``a`` is an iteration variable that in is
each loop associated with a single attribute).  The field ``varType``
contains the type of the attribute; for discrete attributes, ``varType``
is equal to ``orange.VarTypes.Discrete``, and for continuous ``varType`` is
equal to ``orange.VarTypes.Continuous``.

To obtain the number of instances for each class, we first
initialized a vector c that would of the length equal to the number of
different classes. Then, we iterated through the data;
``e.getclass()`` returns a class of an instance e, and to
turn it into a class index (a number that is in range from 0 to n-1,
where n is the number of classes) and is used for an index of a
element of c that should be incremented.

Throughout the code, notice that a print statement in Python prints
whatever items it has in the line that follows. The items are
separated with commas, and Python will by default put a blank between
them when printing. It will also print a new line, unless the print
statement ends with a comma. It is possible to use print statement in
Python with formatting directives, just like in C or C++, but this is
beyond this text.

Running the above script, we obtain the following output::

   Classes: 2
   Attributes: 14 , 6 continuous, 8 discrete
   Instances:  977 total , 236 with class >50K , 741 with class <=50K

If you would like class distributions printed as proportions of
each class in the data sets, then the last part of the script needs
to be slightly changed. This time, we have used string formatting
with print as well (part of :download:`data_characteristics2.py <code/data_characteristics2.py>`)::

   # obtain class distribution
   c = [0] * len(data.domain.classVar.values)
   for e in data:
       c[int(e.getclass())] += 1
   print "Instances: ", len(data), "total",
   r = [0.] * len(c)
   for i in range(len(c)):
       r[i] = c[i]*100./len(data)
   for i in range(len(data.domain.classVar.values)):
       print ", %d(%4.1f%s) with class %s" % (c[i], r[i], '%', data.domain.classVar.values[i]),
   print

The new script outputs the following information::

   Classes: 2
   Attributes: 14 , 6 continuous, 8 discrete
   Instances:  977 total , 236(24.2%) with class >50K , 741(75.8%) with class <=50K

As it turns out, there are more people that earn less than those,
that earn more... On a more technical site, such information may
be important when your build your classifier; the base error for this
data set is 1-.758 = .242, and your constructed models should only be
better than this.

Contingency matrix
------------------

.. index::
   single: basic data exploration; class distribution

Another interesting piece of information that we can obtain from the
data is the distribution of classes for each value of the discrete
attribute, and means for continuous attribute (we will leave the
computation of standard deviation and other statistics to you). Let's
compute means of continuous attributes first (part of :download:`data_characteristics3.py <code/data_characteristics3.py>`)::

   print "Continuous attributes:"
   for a in range(len(data.domain.attributes)):
       if data.domain.attributes[a].varType == orange.VarTypes.Continuous:
           d = 0.; n = 0
           for e in data:
               if not e[a].isSpecial():
                   d += e[a]
                   n += 1
           print "  %s, mean=%3.2f" % (data.domain.attributes[a].name, d/n)

This script iterates through attributes (outer for loop), and for
attributes that are continuous (first if statement) computes a sum
over all instances. A single new trick that the script uses is that it
checks if the instance has a defined attribute value.  Namely, for
instance ``e`` and attribute ``a``, ``e[a].isSpecial()`` is true if
the value is not defined (unknown). Variable n stores the number of
instances with defined values of attribute. For our sampled adult data
set, this part of the code outputs::

   Continuous attributes:
     age, mean=37.74
     fnlwgt, mean=189344.06
     education-num, mean=9.97
     capital-gain, mean=1219.90
     capital-loss, mean=99.49
     hours-per-week, mean=40.27
   
For nominal attributes, we could now compose a code that computes,
for each attribute, how many times a specific value was used for each
class. Instead, we used a build-in method DomainContingency, which
does just that. All that our script will do is, mainly, to print it
out in a readable form (part of :download:`data_characteristics3.py <code/data_characteristics3.py>`)::

   print "\nNominal attributes (contingency matrix for classes:", data.domain.classVar.values, ")"
   cont = orange.DomainContingency(data)
   for a in data.domain.attributes:
       if a.varType == orange.VarTypes.Discrete:
           print "  %s:" % a.name
           for v in range(len(a.values)):
               sum = 0
               for cv in cont[a][v]:
                   sum += cv
               print "    %s, total %d, %s" % (a.values[v], sum, cont[a][v])
           print

Notice that the first part of this script is similar to the one that
is dealing with continuous attributes, except that the for loop is a
little bit simpler. With continuous attributes, the iterator in the
loop was an attribute index, whereas in the script above we iterate
through members of ``data.domain.attributes``, which are objects that
represent attributes. Data structures that may be addressed in Orange
by attribute may most often be addressed either by attribute index,
attribute name (string), or an object that represents an attribute.

The output of the code above is rather long (this data set has
some attributes that have rather large sets of values), so we show
only the output for two attributes::

   Nominal attributes (contingency matrix for classes: <>50K, <=50K> )
     workclass:
       Private, total 729, <170.000, 559.000>
       Self-emp-not-inc, total 62, <19.000, 43.000>
       Self-emp-inc, total 22, <10.000, 12.000>
       Federal-gov, total 27, <10.000, 17.000>
       Local-gov, total 53, <14.000, 39.000>
       State-gov, total 39, <10.000, 29.000>
       Without-pay, total 1, <0.000, 1.000>
       Never-worked, total 0, <0.000, 0.000>
   
     sex:
       Female, total 330, <28.000, 302.000>
       Male, total 647, <208.000, 439.000>

First, notice that the in the vectors the first number refers to a
higher income, and the second number to the lower income (e.g., from
this data it looks like that women earn less than men). Notice that
Orange outputs the tuples. To change this, we would need another loop
that would iterate through members of the tuples. You may also foresee
that it would be interesting to compute the proportions rather than
number of instances in above contingency matrix, but that we leave for
your exercise.

Missing values
--------------

.. index::
   single: missing values; statistics

It is often interesting to see, given the attribute, what is the
proportion of the instances with that attribute unknown. We have
already learned that if a function isSpecial() can be used to
determine if for specific instances and attribute the value is not
defined. Let us use this function to compute the proportion of missing
values per each attribute (:download:`report_missing.py <code/report_missing.py>`)::

   import orange
   data = orange.ExampleTable("adult_sample")
   
   natt = len(data.domain.attributes)
   missing = [0.] * natt
   for i in data:
       for j in range(natt):
           if i[j].isSpecial():
               missing[j] += 1
   missing = map(lambda x, l=len(data):x/l*100., missing)
   
   print "Missing values per attribute:"
   atts = data.domain.attributes
   for i in range(natt):
       print "  %5.1f%s %s" % (missing[i], '%', atts[i].name)

Integer variable natt stores number of attributes in the data set. An
array missing stores the number of the missing values per attribute;
its size is therefore equal to natt, and all of its elements are
initially 0 (in fact, 0.0, since we purposely identified it as a real
number, which helped us later when we converted it to percents).

The only line that possibly looks (very?) strange is ``missing =
map(lambda x, l=len(data):x/l*100., missing)``. This line could be
replaced with for loop, but we just wanted to have it here to show how
coding in Python may look very strange, but may gain in
efficiency. The function map takes a vector (in our case missing), and
executes a function on every of its elements, thus obtaining a new
vector. The function it executes is in our case defined inline, and is
in Python called lambda expression. You can see that our lambda
function takes a single argument (when mapped, an element of vector
missing), and returns its value that is normalized with the number of
data instances (``len(data)``) multiplied by 100, to turn it in
percentage. Thus, the map function in fact normalizes the elements of
missing to express a proportion of missing values over the instances
of the data set.

Finally, let us see what outputs the script we have just been working
on::

   Missing values per attribute:
       0.0% age
       4.5% workclass
       0.0% fnlwgt
       0.0% education
       0.0% education-num
       0.0% marital-status
       4.5% occupation
       0.0% relationship
       0.0% race
       0.0% sex
       0.0% capital-gain
       0.0% capital-loss
       0.0% hours-per-week
       1.9% native-country

In our sampled data set, just three attributes contain the missing
values.

Distributions of feature values
-------------------------------

For some of the tasks above, Orange can provide a shortcut by means of
``orange.DomainDistributions`` function which returns an object that
holds averages and mean square errors for continuous attributes, value
frequencies for discrete attributes, and for both number of instances
where specific attribute has a missing value.  The use of this object
is exemplified in the following script (:download:`data_characteristics4.py <code/data_characteristics4.py>`)::

   import orange
   data = orange.ExampleTable("adult_sample")
   dist = orange.DomainDistributions(data)
   
   print "Average values and mean square errors:"
   for i in range(len(data.domain.attributes)):
       if data.domain.attributes[i].varType == orange.VarTypes.Continuous:
           print "%s, mean=%5.2f +- %5.2f" % \
               (data.domain.attributes[i].name, dist[i].average(), dist[i].error())
   
   print "\nFrequencies for values of discrete attributes:"
   for i in range(len(data.domain.attributes)):
       a = data.domain.attributes[i]
       if a.varType == orange.VarTypes.Discrete:
           print "%s:" % a.name
           for j in range(len(a.values)):
               print "  %s: %d" % (a.values[j], int(dist[i][j]))
   
   print "\nNumber of items where attribute is not defined:"
   for i in range(len(data.domain.attributes)):
       a = data.domain.attributes[i]
       print "  %2d %s" % (dist[i].unknowns, a.name)

Check this script out. Its results should match with the results we
have derived by other scripts in this lesson.
