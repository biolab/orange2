Data input
==========

.. index:: data input

Orange is a machine learning and data mining suite, so
loading-in the data is, as you may acknowledge, its essential
functionality (we tried not to stop here, though, so read on).
Orange supports C4.5, Assistant, Retis, and tab-delimited (native
Orange) data formats. Of these, you may be most familiar with C4.5,
so we will say something about this here, whereas Orange's
native format is the simplest so most of our data files will come
in this flavor.

Let us start with example and Orange native data format. Let us
consider an artificial data set :download:`lenses.tab <code/lenses.tab>` on prescription of eye
lenses [CJ1987]. The data set has four attributes (age of the patient,
spectacle prescription, notion on astigmatism, and information on tear
production rate) plus an associated three-valued class, that gives the
appropriate lens prescription for patient (hard contact lenses, soft
contact lenses, no lenses). You may have already guessed that this
data set can, in principle, be used to build a classifier that, based
on the four attributes, prescribes the right lenses. But before we do
that, let us see how the data set file is composed and how to read it
in Orange by displaying first few lines of :download:`lenses.tab <code/lenses.tab>`::

   age       prescription  astigmatic    tear_rate     lenses
   discrete  discrete      discrete      discrete      discrete 
                                                       class
   young     myope         no            reduced       none
   young     myope         no            normal        soft
   young     myope         yes           reduced       none
   young     myope         yes           normal        hard
   young     hypermetrope  no            reduced       none

First line of the file lists names of attributes and class.
Second line gives the type of the attribute. Here, all attributes
are nominal (or discrete), hence the ``discrete`` keyword
any every column. If you get tired of typing
``discrete``, you may use ``d`` instead. We
will later find that attribute may also be continuous, and will
have appropriate keyword (or just ``c``) in their
corresponding columns. The third line adds an additional
description to every column. Note that ``lenses`` is a
special variable since it represents a class where each data
instance is classified. This is denoted as ``class`` in
the third line of the last column. Other keywords may be used in
this line that we have not used in our example. For instance, for
the attributes that we would like to ignore, we can use
``ignore`` keyword (or simply ``i``). There are
also other keywords that may be used, but for the sake of
simplicity we will skip all this here.

The rest of the table gives the data. Note that there are 5
instances in our table above (check the original file to see
other). Orange is rather free in what attribute value names it
uses, so they do not need all to start with a letter like in our
example.

Attribute values are separated with tabulators (<TAB>).  This is
rather hard to see above (it looks like spaces were used), so to
verify that check the original :download:`lenses.tab <code/lenses.tab>` data set in
your favorite text editor.  Alternatively, authors of this text like
best to edit these files in a spreadsheet program (and use
tab-delimited format to save the files), so a snapshot of the data set
as edited in Excel can look like this:

.. image:: files/excel.png
   :alt: Data in Excel

Now create a directory, save :download:`lenses.tab <code/lenses.tab>` in
it (right click on the link and choose choose "Save Target As
..."). Open a terminal (cmd shell in Windows, Terminal on Mac OS X),
change the directory to the one you have just created, and run
Python. In the interactive Python shell, import Orange and the data
file:

>>> import orange
>>> data = orange.ExampleTable("lenses")
>>>

This creates an object called data that holds your data set and
information about the lenses domain. Note that for the file name no
suffix was needed: Orange ventures through the current directory
and checks if any files of the types it knows are available. This
time, it found lenses.tab.

How do we know that data really contains our data set? Well,
let's check this out and print the attribute names and first
3 data items:

>>> print data.domain.attributes
<age, prescription, astigmatic, tear_rate>
>>> for i in range(3):
... 	print data[i]
... 	
['young', 'myope', 'no', 'reduced', 'none']
['young', 'myope', 'no', 'normal', 'soft']
['young', 'myope', 'yes', 'reduced', 'none']
>>>

Now let's put together a script file :download:`lenses.py <code/lenses.py>` that
reads lenses data, prints out names of the attributes and class, and
lists first 5 data instances (:download:`lenses.py <code/lenses.py>`)::

   import orange
   data = orange.ExampleTable("lenses")
   print "Attributes:",
   for i in data.domain.attributes:
       print i.name,
   print
   print "Class:", data.domain.classVar.name
   
   print "First 5 data items:"
   for i in range(5):
      print data[i]

Few comments on this script are in place. First, note that data
is an object that holds both the data and information on the
domain. We show above how to access attribute and class names, but
you may correctly expect that there is much more information there,
including on attribute type, values it may hold, etc. Also notice
the particular syntax python uses for ``for`` loops: the
line that declares the loop ends with ``:``, and whatever
is in the loop is indented (we have used three spaces to indent the
statements that are within each loop).

Save :download:`lenses.py <code/lenses.py>` in your working directory. There
should now be both files lenses.py and lenses.tab. Now let's see if we
run the script we have just written::

   > python lenses.py
   Attributes: age prescription astigmatic tear_rate
   Class: lenses
   First 5 data items:
   ['young', 'myope', 'no', 'reduced', 'none']
   ['young', 'myope', 'no', 'normal', 'soft']
   ['young', 'myope', 'yes', 'reduced', 'none']
   ['young', 'myope', 'yes', 'normal', 'hard']
   ['young', 'hypermetrope', 'no', 'reduced', 'none']
   >

Now, we promised to say something about C4.5 data files, which syntax
was (and perhaps still is) common within machine learning community
due to extensive use of this program. Notice that C4.5 data sets are
described within two files: file with extension ".data" holds the
actual data, whereas domain (attribute and class names and types) are
described in a separate file ".names".  Instead of going into how
exactly these files are formed, we show just an example that Orange
can handle them. For this purpose, load :download:`car.data <code/car.data>` and
:download:`car.names <code/car.names>` and run the following code::

   > python
   >>> car_data = orange.ExampleTable("car")
   >>> print car_data.domain.attributes
   <buying, maint, doors, persons, lugboot, safety>
   >>>

If you think that storing domain information and data in a single
file, or if you better like looking to your data through the
spreadsheet, you may now store your C4.5 data file to a Orange native
(.tab) format:

>>> orange.saveTabDelimited ("car.tab", car_data)
>>>

Similarly, saving to C4.5 format is possible through ``orange.saveC45``.

Above all applies if you run Python through Command Prompt. If you use
PythonWin, however, you have to tell it where exactly your data is
located. You may either need to specify absolute path of your data
files, like (type your commands in Interactive Window):

>>> car_data = orange.ExampleTable("c:/orange/car")
>>>

or set a working directory through Python's os library:

>>> import os
>>> os.chdir("c:/orange")
>>>

**References**

.. [CJ1987] Cendrowska J (1987) PRISM: An algorithm for inducing modular rules,
   International Journal of Man-Machine Studies, 27, 349-370.
