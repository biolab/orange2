################################
Writing Orange Extensions in C++
################################

This page gives an introduction to extending Orange in C++ with emphasis on
how to define interfaces to Python. Besides reading this page, we recommend
studying some of existing extension modules like orangeom, and the Orange's
interface itself.

We shall first present a general picture and then focus on specific parts of the
interface.

Instead of general tools for creating interfaces between C++ and Python
(Swig, Sip, PyBoost...), Orange uses its own specific set of tools.

To expose a C++ object to Python, we need to mark them as exportable, select a
general constructor template to use or program a specific one, we have to mark
the attributes to be exported, and provide the interfaces for C++ member
functions. When we give the access to mostly C++ code as it is, the interface
functions have only a few lines. When we want to make the exported function more
friendly, eg. allow various types of arguments or fitting the default arguments
according to the given ones, these functions are longer.

To define a non-member function, we write the function itself as described in
the Python's manual (see the first chapter of "Extending and Embedding the
Python Interpreter") and then mark it with a specific keyword.
Pyxtract will recognize the keyword and add it to the list of exported functions.

To define a special method, one needs to provide a function with the appropriate
name constructed from the class name and the special method's name, which is the
same as in Python's PyTypeObjects.

For instance, the elements of ``ExampleTable`` (examples) can be accessed
through indexing because we defined a C function that gets an index (and the
table, of course) and returns the corresponding example. Here is the function
(with error detection removed for the sake of clarity). ::

    PyObject *ExampleTable_getitem_sq(PyObject *self, int idx)
    {
        CAST_TO(TExampleTable, table);
        return Example_FromExampleRef((*table)[idx], EXAMPLE_LOCK(PyOrange_AsExampleTable(self)));
    }

Also, ``ExampleTable`` has a non-special method ``sort([list-of-attributes])``.
This is implemented through a C function that gets a list of attributes and
calls the C++ class' method
``TExampleTable::sort(const vector<int> order)``. To illustrate, this is a
slightly simplified function (we've removed some flexibility regarding the
parameters and the exception handling). ::

    PyObject *ExampleTable_sort(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")
    {
        CAST_TO(TExampleTable, table);

        if (!args || !PyTuple_Size(args)) {
            table->sort();
            RETURN_NONE;
        }

        TVarList attributes;
        varListFromDomain(PyTuple_GET_ITEM(args, 0), table->domain, attributes, true, true);
        vector<int> order;
        for(TVarList::reverse_iterator vi(attributes.rbegin()), ve(attributes.rend()); vi!=ve; vi++) {
            order.push_back(table->domain->getVarNum(*vi));
        }
        table->sort(order);
        RETURN_NONE;
    }

The function casts the ``PyObject *`` into the
corresponding C++ object, reads the arguments, calls the C++
functions and returns the result (``None``, in this case).

Interfacing with Python requires a lot of manual work, but this gives a
programmer the opportunity to provide a function which accepts many different
forms of arguments. The above function, for instance, accepts a list in
which attributes are specified by indices, names or descriptors, all
corresponding to the ``ExampleTable`` which is being sorted. Inheritance of
methods, on the other hand, ensures that only the methods that are truly
specific for a class need to be coded.

The part of the interface that is built automatically is taken care of by
two scripts. ``pyprops`` parses all Orange's header files and extracts all
the class built-in properties. The second is ``pyxtract``, which goes
through the C++ files that contain the interface functions such as those above.
It recognizes the functions that implement special or member methods and
constructs the corresponding ``PyTypeObject``s.

*******
pyprops
*******

Pyprops scans each hpp file for classes we want to export to Python). Properties
can be ``bool``, ``int``, ``float``, ``string``, ``TValue`` or a wrapped Orange
type.

Class definition needs to look as follows. ::

    class [ORANGE_API] <classname>; [: public <parentclass> ]

This should be in a single line. To mark the class for export, this should be
followed by ``__REGISTER_CLASS`` or ``__REGISTER_ABSTRACT_CLASS`` before any
properties or components are defined. The difference between the two, as far as
pyprops is concerned, is that abstract classes do not define the ``clone``
method.

To export a property, it should be defined like this. ::

    <type> <name> //P[R|O] [>|+<alias>] <description>

Pyprops doesn't check the type and won't object if you use other types than
those listed above. The error will be discovered later, during linking. ``//P``
signals that we want to export the property. If followed by ``R`` or ``O``, the
property is read-only or obsolete. The property can also have an alias name;
``>`` renames it and ``+`` adds an alias.

Each property needs to be declared in a separate line, e.g. ::

    int x; //P;
    int y; //P;

If we don't want to export a certain property, we omit the ``//P`` mark. An
exception to this are wrapped Orange objects: for instance, if a class has a
(wrapped) pointer to the domain, ``PDomain`` and it doesn't export it, pyxtract
should still know about them because for the purpose of garbage collection. You
should mark them by ``//C`` so that they are put into the list of objects that
need to be counted. Failing to do so would cause a memory leak.

If a class directly or indirectly holds references to any wrapped objects that
are neither properties nor components, it needs to declare ``traverse`` and
``clear`` as described in Python documentation.

Pyprops creates a ppp file for each hpp, which includes the extracted
information in form of C++ structures that compile into the interface.
The ppp file needs to be included in the corresponding cpp file. For
instance, domain.ppp is included in domain.cpp.

********
pyxtract
********

Pyxtract's job is to detect the functions that define special methods (such as
printing, conversion, sequence and arithmetic related operations...) and member
functions. Based on what it finds for each specific class, it constructs the
corresponding ``PyTypeObject``s. For the functions to be recognized, they must
follow a specific syntax.

There are two basic mechanisms for marking the functions to export. Special
functions are recognized by their definition (they need to return
``PyObject *``, ``void`` or ``int`` and their name must be of form
<classname>_<functionname>). Member functions,
inheritance relations, constants etc. are marked by macros such as ``PYARGS``
in the above definition of ``ExampleTable_sort``. Most of these macros don't do
anything except for marking stuff for pyxtract.

Class declaration
=================

Each class needs to be declared as exportable. If it's a base class, pyxtract
needs to know the data structure for the instances of this class. As for all
Python objects the structure must be "derived" from ``PyObject`` (Python is
written in C, so the subclasses are not derived in the C++ sense but extend the
C structure instead). Most objects are derived from Orange; the only exceptions
are ``orange.Example``, ``orange.Value`` and ``orange.DomainDepot``.

Pyxtract should also know how the class is constructed - it can have a specific
constructor, one of the general constructors or no constructor at all.

The class is declared in one of the following ways (here are some examples from
actual Orange code).

``BASED_ON(EFMDataDescription, Orange)``
    This tells pyxtract that ``EFMDataDescription`` is an abstract class derived from ``Orange``: there is no constructor for this class in Python, but the C++ class itself is not abstract and can appear and be used in Python. For example, when we construct an instance of ``ClassifierByLookupTable`` with more than three attributes, an instance of ``EFMDataDescription`` will appear in one of its fields.

``ABSTRACT(ClassifierFD, Classifier)``
    This defines an abstract class, which will never be constructed in the C++ code. The only difference between this ``BASED_ON`` and ``ABSTRACT`` is that the former can have pickle interface, while the latter don't need one.

Abstract C++ classes are not necessarily defined as ``ABSTRACT`` in the Python
interface. For example, ``TClassifier`` is an abstract C++ class, but you can
seemingly construct an instance of ``Classifier`` in Python. What happens is
that there is an additional C++ class ``TClassifierPython``, which poses as
Python's class ``Classifier``. So the Python class ``Classifier`` is not defined
as ``ABSTRACT`` or ``BASED_ON`` but using the ``Classifier_new`` function, as
described below.


``C_NAMED(EnumVariable, Variable, "([name=, values=, autoValues=, distributed=, getValueFrom=])")``
    ``EnumVariable`` is derived from ``Variable``. Pyxtract will also create a constructor which will accept the object's name as an optional argument. The third argument is a string that describes the constructor, eg. gives a list of arguments. IDEs for Python, such as PythonWin, will show this string in a balloon help while the programmer is typing.

``C_UNNAMED(RandomGenerator, Orange, "() -> RandomGenerator")``
    This is similar as ``C_NAMED``, except that the constructor accepts no name. This form is rather rare since all Orange objects can be named.

``C_CALL(BayesLearner, Learner, "([examples], [weight=, estimate=] -/-> Classifier")``
    ``BayesLearner`` is derived from ``Learner``. It will have a peculiar constructor. It will, as usual, first construct an instance of ``BayesLearner``. If no arguments are given (except for, possibly, keyword arguments), it will return the constructed instance. Otherwise, it will call the ``Learner``'s call operator and return its result instead of ``BayesLearner``.

``C_CALL3(MakeRandomIndices2, MakeRandomIndices2, MakeRandomIndices, "[n | gen [, p0]], [p0=, stratified=, randseed=] -/-> [int]")``
    ``MakeRandomIndices2`` is derived from ``MakeRandomIndices`` (the third argument). For a contrast from the ``C_CALL`` above, the corresponding constructor won't call ``MakeRandomIndices`` call operator, but the call operator of ``MakeRandomIndices2`` (the second argument). This constructor is often used when the parent class doesn't provide a suitable call operator.

``HIDDEN(TreeStopCriteria_Python, TreeStopCriteria)``
    ``TreeStopCriteria_Python`` is derived from ``TreeStopCriteria``, but we would like to hide this class from the user. We use this definition when it is elegant for us to have some intermediate class or a class that implements some specific functionality, but don't want to bother the user with it. The class is not completely hidden - the user can reach it through the ``type`` operator on an instance of it. This is thus very similar to a ``BASED_ON``.

``DATASTRUCTURE(Orange, TPyOrange, orange_dict)``
    This is for the base classes. ``Orange`` has no parent class. The C++ structure that stores it is ``TPyOrange``; ``TPyOrange`` is essentially ``PyObject`` (again, the structure always has to be based on ``PyObject``) but with several additional fields, among them a pointer to an instance of ``TOrange`` (the C++ base class for all Orange's classes). ``orange_dict`` is a name of ``TPyOrange``'s field that points to a Python dictionary; when you have an instance ``bayesClassifier`` and you type, in Python, ``bayesClassifier.someMyData=15``, this gets stored in ``orange_dict``. The actual mechanism behind this is rather complicated and you most probably won't need to use it. If you happen to need to define a class with ``DATASTRUCTURE``, you can simply omit the last argument and give a 0 instead.

Even if the class is defined by ``DATASTRUCTURE``, you can still specify a
different constructor, most probably the last form of it (the ``_new``
function). In this case, specify a keyword ``ROOT`` as a parent and pyxtract
will understand that this is the base class.

Object construction in Python is divided between two methods. The constructors
we discussed above construct the essential part of the object - they allocate
the necessary memory and initialize the fields far enough that the object is
valid to enter the garbage collection. The second part is handled by the
``init`` method. It is, however, not forbidden to organize the things so that
``new`` does all the job. This is also the case in Orange. The only task left
for ``init`` is to set any attributes that user gave as the keyword arguments to
the constructor.

For instance, Python's statement
``orange.EnumVariable("a", values=["a", "b", "c"])`` is executed so that ``new``
constructs the variable and gives it the name, while ``init`` sets the
``values`` field.

The ``new`` operator can also accept keyword arguments. For
instance, when constructing an ``ExampleTable`` by reading the data from a file,
you can specify a domain (using keyword argument ``domain``), a list of
attributes to reuse if possible (``use``), you can tell it not to reuse the
stored domain or not to store the newly constructed domain (``dontCheckStored``,
``dontStore``). After the ``ExampleTable`` is constructed, ``init`` is called to
set the attributes. To tell it to ignore the keyword arguments that the
constructor might (or had) used, we write the following. ::

    CONSTRUCTOR_KEYWORDS(ExampleTable, "domain use useMetas dontCheckStored dontStore filterMetas")

There's another macro related to attributes. Let ``ba`` be an orange object, say
an instance of ``orange.BayesLearner``. If you assign new attributes as usual
directly, eg. ``ba.myAttribute = 12``, you will get a warning (you should use
the object's method ``setattr(name, value)`` to avoid it). Some objects have
some attributes that cannot be implemented in C++ code, yet they are usual and
useful. For instance, ``Graph`` can use attributes ``objects``, ``forceMapping``
and ``returnIndices``, which can only be set from Python (if you take a look at
the documentation on ``Graph`` you will see why these cannot be implemented in
C++). Yet, since user are allowed to set these attributes and will do so often,
we don't want to give warnings. We achieve this by ::

    RECOGNIZED_ATTRIBUTES(Graph, "objects forceMapping returnIndices")


Special methods
===============

Special methods act as the class built-in methods. They define what the type can
do: if it, for instance, supports multiplication, it should define the operator
that gets the object itself and another object and return the product (or throw
an exception). If it allows for indexing, it defines an operator that gets the
object itself and the index, and returns the element. These operators are
low-level; most can be called from Python scripts but they are also internally
by Python. For instance, if ``table`` is an ``ExampleTable``, then
``for e in table:`` or ``reduce(f, table)`` will both work by calling the
indexing operator for each table's element.
For more details, consider the Python manual, chapter "Extending and
Embedding the Python Interpreter" section "Defining New Types".

To define a method for Orange class, you need to define a function named,
``<classname>_<methodname>``; the function should return either
``PyObject *``, ``int`` or ``void``. The function's head has to be written in a
single line. Regarding the arguments and the result, it should conform to
Python's specifications. Pyxtract will detect the methods and set the pointers
in ``PyTypeObject`` correspondingly.

Here is a list of methods: the left column represents a method name that
triggers pyxtract (these names generally correspond to special method names of
Python classes as a programmer in Python sees them) and the second is the
name of the field in ``PyTypeObject`` or subjugated structures. See Python
documentation for description of functions' arguments and results. Not all
methods can be directly defined; for those that can't, it is because we either
use an alternative method (eg. ``setattro`` instead of ``setattr``) or pyxtract
gets or computes the data for this field in some other way.

General methods
---------------

+--------------+-----------------------+-----------------------------------------------------------+
| pyxtract     | PyTypeObject          |                                                           |
+==============+=======================+===========================================================+
| ``dealloc``  | ``tp_dealloc``        | Frees the memory occupied by the object. You will need to |
|              |                       | define this for the classes with a new ``DATASTRUCTURE``; |
|              |                       | if you only derive a class from some Orange class, this   |
|              |                       | has been taken care of. If you have a brand new object,   |
|              |                       | copy the code of one of Orange's deallocators.            |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_getattr``        | Can't be redefined since we use ``tp_getattro`` instead.  |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_setattr``        | Can't be redefined since we use ``tp_setattro`` instead.  |
+--------------+-----------------------+-----------------------------------------------------------+
| ``cmp``      | ``tp_compare``        |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``repr``     | ``tp_repr``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``as_number``         | (pyxtract will initialize this field if you give any of   |
|              |                       | the methods from the number protocol; you needn't care    |
|              |                       | about this field)                                         |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``as_sequence``       | (pyxtract will initialize this field if you give any of   |
|              |                       | the methods from the sequence protocol)                   |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``as_mapping``        | (pyxtract will initialize this field if you give any of   |
|              |                       | the methods from the mapping protocol)                    |
+--------------+-----------------------+-----------------------------------------------------------+
| ``hash``     | ``tp_hash``           | Class ``Orange`` computes a hash value from the pointer;  |
|              |                       | you don't need to overload it if your object inherits the |
|              |                       | function. If you write an independent class, just copy the|
|              |                       | code.                                                     |
+--------------+-----------------------+-----------------------------------------------------------+
| ``call``     | ``tp_call``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``call``     | ``tp_call``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``str``      | ``tp_str``            |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``getattr``  | ``tp_getattro``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``setattr``  | ``tp_setattro``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_as_buffer``      | Pyxtract doesn't support the buffer protocol.             |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_flags``          | Flags are set by pyxtract.                                |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_doc``            | Documentation is read from the constructor definition     |
|              |                       | (see above).                                              |
+--------------+-----------------------+-----------------------------------------------------------+
| ``traverse`` | ``tp_traverse``       | Traverse is tricky (as is garbage collection in general). |
|              |                       | There's something on it in a comment in root.hpp; besides |
|              |                       | that, study the examples. In general, if a wrapped member |
|              |                       | is exported to Python (just as, for instance,             |
|              |                       | ``Classifier`` contains a ``Variable`` named              |
|              |                       | ``classVar``), you don't need to care about it. You should|
|              |                       | manually take care of any wrapped objects not exported to |
|              |                       | Python. You probably won't come across such cases.        |
+--------------+-----------------------+-----------------------------------------------------------+
| ``clear``    | ``tp_clear``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``richcmp``  | ``tp_richcmp``        |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_weaklistoffset`` |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``iter``     | ``tp_iter``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``iternext`` | ``tp_iternext``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_methods``        | Set by pyxtract if any methods are given.                 |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_members``        |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``getset``            | Pyxtract initializes this by a pointer to manually        |
|              |                       | written getters/setters (see below).                      |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_base``           | Set by pyxtract to a class specified in constructor       |
|              |                       | (see above).                                              |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_dict``           | Used for class constants (eg. ``Classifier.GetBoth``)     |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_descrget``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_descrset``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_dictoffset``     | Set by pyxtract to the field given in ``DATASTRUCTURE``   |
|              |                       | (if there is any).                                        |
+--------------+-----------------------+-----------------------------------------------------------+
| ``init``     | ``tp_init``           |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_alloc``          | Set to ``PyType_GenericAlloc``                            |
+--------------+-----------------------+-----------------------------------------------------------+
| ``new``      | ``tp_new``            |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_free``           | Set to ``_PyObject_GC_Del``                               |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_is_gc``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_bases``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_mro``            |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_cache``          |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_subclasses``     |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+
| ``.``        | ``tp_weaklist``       |                                                           |
+--------------+-----------------------+-----------------------------------------------------------+

Numeric protocol
----------------

+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``add``    |  ``nb_add``      | ``pow``     | ``nb_power``    | ``lshift`` | ``nb_lshift`` | ``int``   | ``nb_int``   |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``sub``    | ``nb_subtract``  | ``neg``     | ``nb_negative`` | ``rshift`` | ``nb_rshift`` | ``long``  | ``nb_long``  |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``mul``    | ``nb_multiply``  | ``pos``     | ``nb_positive`` | ``and``    | ``nb_and``    | ``float`` | ``nb_float`` |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``div``    | ``nb_divide``    | ``abs``     | ``nb_absolute`` | ``or``     | ``nb_or``     | ``oct``   | ``nb_oct``   |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``mod``    | ``nb_remainder`` | ``nonzero`` | ``nb_nonzero``  | ``coerce`` | ``nb_coerce`` | ``hex``   | ``nb_hex``   |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+
| ``divmod`` | ``nb_divmod``    | ``inv``     | ``nb_invert``   |            |               |           |              |
+------------+------------------+-------------+-----------------+------------+---------------+-----------+--------------+

Sequence protocol
-----------------

+----------------+---------------+----------------+------------------+
| ``len_sq``     | ``sq_length`` | ``getslice``   | ``sq_slice``     |
+----------------+---------------+----------------+------------------+
| ``concat``     | ``sq_concat`` | ``setitem_sq`` | ``sq_ass_item``  |
+----------------+---------------+----------------+------------------+
| ``repeat``     | ``sq_slice``  | ``setslice``   | ``sq_ass_slice`` |
+----------------+---------------+----------------+------------------+
| ``getitem_sq`` | ``sq_item``   | ``contains``   | ``sq_contains``  |
+----------------+---------------+----------------+------------------+

Mapping protocol
----------------

+-------------+----------------------+
| ``len``     | ``mp_length``        |
+-------------+----------------------+
| ``getitem`` | ``mp_subscript``     |
+-------------+----------------------+
| ``setitem`` | ``mp_ass_subscript`` |
+-------------+----------------------+

For example, here is what gets called when you want to know the length of an
example table. ::

    int ExampleTable_len_sq(PyObject *self)
    {
        PyTRY
            return SELF_AS(TExampleGenerator).numberOfExamples();
        PyCATCH_1
    }

``PyTRY`` and ``PyCATCH`` take care of C++ exceptions. ``SELF_AS`` is a macro
for casting, ie unwrapping the points (this is an alternative to ``CAST_TO``).


Getting and Setting Class Attributes
====================================

Exporting of most of C++ class fields is already taken care by the lists that
are compiled by pyprops. There are only a few cases in the entire Orange where
we needed to manually write specific handlers for setting and getting the
attributes. This needs to be done if setting needs some special processing or
when simulating an attribute that does not exist in the underlying C++ class.

An example for this is class ``HierarchicalCluster``. It contains results of a
general, not necessarily binary clustering, so each node in the tree has a list
``branches`` with all the node's children. Yet, as the usual clustering is
binary, it would be nice if the node would also support attributes ``left`` and
``right``. They are not present in C++, but we can write a function that check
the number of branches; if there are none, it returns ``None``, if there are
more than two, it complains, while otherwise it returns the first branch. ::

    PyObject *HierarchicalCluster_get_left(PyObject *self)
    {
        PyTRY
            CAST_TO(THierarchicalCluster, cluster);

            if (!cluster->branches)
                RETURN_NONE

            if (cluster->branches->size() > 2)
                PYERROR(PyExc_AttributeError,
                        "'left' not defined (cluster has more than two subclusters)",
                        NULL);

            return WrapOrange(cluster->branches->front());
        PyCATCH
    }

As you can see from the example, the function needs to accept a ``PyObject *``
(the object it``self``) and return a ``PyObject *`` (the attribute value). The
function name needs to be ``<classname>_get_<attributename>``.
Setting an attribute is similar; function name should be
``<classname>_set_<attributename>``, it should accept two Python
objects (the object and the attribute value) and return an ``int``, where 0
signifies success and -1 a failure.

If you define only one of the two handlers, you'll get a read-only or write-only
attribute.


Member functions
================

We have already shown an example of a member function - the ``ExampleTable``'s
method ``sort``. The general template is
``PyObject *<classname>_<methodname>(<arguments>) PYARGS(<arguments-keyword>, <documentation-string>)``.
In the case of the ``ExampleTable``'s ``sort``, this looks like this. ::

    PyObject *ExampleTable_sort(PyObject *self, PyObject *args) PYARGS(METH_VARARGS, "() -> None")

Argument type can be any of the usual Python constants stating the number and
the kind of arguments, such as ``METH_VARARGS`` or ``METH_O`` - this constant
gets copied to the corresponding list (browse Python documentation for
``PyMethodDef``).


Class constants
===============

Orange classes, as seen from Python, can also have constants, such as
``orange.Classifier.GetBoth``. Classifier's ``GetBoth`` is visible as a member
of the class, the derived classes and all their instances (eg.
``BayesClassifier.GetBoth`` and ``bayes.GetBoth``).

There are several ways to define such constants. If they are simple integers or
floats, you can use ``PYCLASSCONSTANT_INT`` or ``PYCLASSCONSTANT_FLOAT``, like
in ::

    PYCLASSCONSTANT_INT(Classifier, GetBoth, 2)

You can also use the enums from the class, like ::

    PYCLASSCONSTANT_INT(C45TreeNode, Leaf, TC45TreeNode::Leaf)

Pyxtract will convert the given constant to a Python object (using
``PyInt_FromLong`` or ``PyFloat_FromDouble>``).

When the constant is an object of some other type, use ``PYCLASSCONSTANT``. In
this form (not used in Orange so far), the third argument can be either an
instance of ``PyObject *`` or a function call. In either case, the object or
function must be known at the point where the pyxtract generated file is
included.


Pickling
========

Pickling is taken care of automatically if the class provides a Python
constructor that can construct the object without arguments (it may *accept*
arguments, but should be able to do without them. If there is no such
constructor, the class should provide a ``__reduce__`` method or it should
explicitly declare that it cannot be pickled. If it doesn't pyxtract will issue
a warning that the class will not be picklable.

Here are the rules:

* Classes that provide a ``__reduce__`` method (details follow below) are pickled through that method.

* Class ``Orange``, the base class, already provides a ``__reduce__`` method, which is only useful if the constructor accepts empty arguments. So, if the constructor is declared as ``C_NAMED``, ``C_UNNAMED``, ``C_CALL`` or ``C_CALL3``, the class is the class will be picklable. See the warning below.

* If the constructor is defined by ``_new`` method, and the ``BASED_ON`` definition is followed be ``ALLOWS_EMPTY``, this signifies that it accepts empty arguments, so it will be picklable just as in the above point. For example, the constructor for the class ``DefaultClassifier`` is defined like this ::

    PyObject *DefaultClassifier_new(PyTypeObject *tpe, PyObject *args)
        BASED_ON(Classifier, "([defaultVal])") ALLOWS_EMPTY

and is picklable through code ``Orange.__reduce__``. But again, see the warning
below.

* If the constructor is defined as ``ABSTRACT``, there cannot be any instances of this class, so pyxtract will give no warning that it is not picklable.
* The class can be explicitly defined as not picklable by ``NO_PICKLE`` macro, as in ::

    NO_PICKLE(TabDelimExampleGenerator)

  Such classes won't be picklable even if they define the appropriate
  constructors. This effectively defined a ``__reduce__`` method which yields an
  exception; if you manually provide a ``__reduce__`` method for such a class,
  pyxtract will detect that the method is multiply defined.

* If there are no suitable constructors, no ``__reduce__`` method and no
  ``ABSTRACT`` or ``NO_PICKLE`` flag, pyxtract gives a warning about that.

When the constructor is used, as in points 2 and 3, pickling will only work if
all fields of the C++ class can be set "manually" from Python, are set through
the constructor, or are set when assigning other fields. In other words, if
there are fields that are not
marked as ``//P`` for pyprops, you will most probably need to manually define
a ``__reduce__`` method, as in point 1.

The details of what the ``__reduce__`` method must do are described in the
Python documentation. In our circumstances, it can be implemented in two ways
which differ in what function is used for unpickling: it can either use the
class' constructor or we can define a special method for unpickling.

The former usually happens when the class has a read-only property (``//PR``),
which is set by the constructor. For instance, ``AssociationRule`` has read-only
fields ``left`` and ``right``, which are needs to be given to the constructor.
This is the ``__reduce__`` method for the class. ::

    PyObject *AssociationRule__reduce__(PyObject *self)
    {
        PyTRY
            CAST_TO(TAssociationRule, arule);
            return Py_BuildValue("O(NN)N", self->ob_type,
                                       Example_FromWrappedExample(arule->left),
                                       Example_FromWrappedExample(arule->right),
                                       packOrangeDictionary(self));
        PyCATCH
    }

As described in the Python documentation, the ``__reduce__`` should return a
tuple in which the first element is the function that will do the unpickling,
and the second argument are the arguments for that function. Our unpickling
function is simply the classes' type (calling a type corresponds to calling a
constructor) and the arguments for the constructor are the left- and right-hand
side of the rule. The third element of the tuple is classes' dictionary.

When unpickling is more complicated - usually when the class has no constructor
and contains fields of type ``float *`` or similar - we need a special
unpickling function. The function needs to be directly in the modules' namespace
(it cannot be a static method of a class), so we named them
``__pickleLoader<classname>``. Search for examples of such functions in
the source code; note that the instance's true class need to be pickled, too.
Also, check how we use ``TCharBuffer`` throughout the code to store and pickle
binary data as Python strings.

Be careful when manually writing the unpickler: if a C++ class derived from that
class inherits its ``__reduce__``, the corresponding unpickler will construct an
instance of a wrong class (unless the unpickler functions through Python's
constructor, ``ob_type->tp_new``). Hence, classes derived from a class which
defines an unpickler have to define their own ``__reduce__``, too.

Non-member functions and constants
==================================

Non-member functions are defined in the same way as member functions except
that their names do not start with the class name. Here is how the ``newmetaid``
is implemented ::

    PyObject *newmetaid(PyObject *, PyObject *) PYARGS(0,"() -> int")
    {
        PyTRY
            return PyInt_FromLong(getMetaID());
        PyCATCH
    }

Orange also defines some non-member constants. These are defined in a similar
fashion as the class constants.
``PYCONSTANT_INT(<constant-name>, <integer>)`` defines an integer
constant and ``PYCONSTANT_FLOAT`` would be used for a continuous one.
``PYCONSTANT`` is used for objects of other types, as the below example that
defines an (obsolete) constant ``MeasureAttribute_splitGain`` shows. ::

    PYCONSTANT(MeasureAttribute_splitGain, (PyObject *)&PyOrMeasureAttribute_gainRatio_Type)

Class constants from the previous section are put in a pyxtract generated file
that is included at the end of the file in which the constant definitions and
the corresponding classes are. Global constant modules are included in another
file, far away from their actual definitions. For this reason, ``PYCONSTANT``
cannot refer to any functions (the above example is an exception - all class
types are declared in this same file and are thus available at the moment the
above code is used). Therefore, if the constant is defined by a function call,
you need to use another keyword, ``PYCONSTANTFUNC``::

    PYCONSTANTFUNC(globalRandom, stdRandomGenerator)

Pyxtract will generate a code which will, prior to calling
``stdRandomGenerator``, declare it as a function with no arguments that returns
``PyObject *``. Of course, you will have to define the function somewhere in
your code, like this::

    PyObject *stdRandomGenerator()
    {
        return WrapOrange(globalRandom);
    }

Another example are ``VarTypes``. ``VarTypes`` is a tiny module inside Orange
that contains nothing but five constants, representing various attribute types.
From pyxtract perspective, ``VarTypes`` is a constant. This is the complete
definition. ::

    PyObject *VarTypes()
    {
        PyObject *vartypes=PyModule_New("VarTypes");
        PyModule_AddIntConstant(vartypes, "None", (int)TValue::NONE);
        PyModule_AddIntConstant(vartypes, "Discrete", (int)TValue::INTVAR);
        PyModule_AddIntConstant(vartypes, "Continuous", (int)TValue::FLOATVAR);
        PyModule_AddIntConstant(vartypes, "Other", (int)TValue::FLOATVAR+1);
        PyModule_AddIntConstant(vartypes, "String", (int)STRINGVAR);
        return vartypes;
    }

    PYCONSTANTFUNC(VarTypes, VarTypes)

If you want to understand the constants completely, check the Orange's pyxtract
generated file initialization.px.

How does it all fit together
============================

We will finish the section with a description of the files generated by the two
scripts. Understanding these may be needed for debugging purposes.

File specific px files
----------------------

For each compiled cpp file, pyxtract creates a px file with the same name. The
file starts with externs declaring the base classes for the classes whose types
are defined later on. Then follow class type definitions:

* Method definitions (``PyMethodDef``). Nothing exotic here, just a table with
  the member functions that is pointed to by ``tp_methods`` of the
  ``PyTypeObject``.

* GetSet definitions (``PyGetSetDef``). Similar to methods, a list to be pointed
  to by ``tp_getset``, which includes the attributes for which special handlers
  were written.

* Definitions of doc strings for call operator and constructor.

* Constants. If the class has any constants, there will be a function named
  ``void <class-name>_addConstants()``. The function will create a class
  dictionary in the type's ``tp_dict``, if there is none yet. Then it will store
  the constants in it. The functions is called at the module initialization,
  file initialization.px.

* Constructors. If the class uses generic constructors (ie, if it's defined by
  ``C_UNNAMED``, ``C_NAMED``, ``C_CALL`` or ``C_CALL3``), they will need to call
  a default object constructor, like the below one for ``FloatVariable``.
  (This supposes the object is derived from ``TOrange``! We will need to get rid
  of this we want pyxtract to be more general. Maybe an additional argument in
  ``DATASTRUCTURE``?) ::

    POrange FloatVariable_default_constructor(PyTypeObject *type)
    {
        return POrange(mlnew TFloatVariable(), type);
    }

  If the class is abstract, pyxtract defines a constructor that will call
  ``PyOrType_GenericAbstract``. ``PyOrType_GenericAbstract`` checks the type
  that the caller wishes to construct; if it is a type derived from this type,
  it permits it, otherwise it complains that the class is abstract.

* Aliases. A list of renamed attributes.

* ``PyTypeObject`` and the numeric, sequence and mapping protocols.
  ``PyTypeObject`` is named ``PyOr<classname>_Type_inh``.

* Definition of conversion functions. This is done by macro
  ``DEFINE_cc(<classname>)`` which defines
  ``int ccn_<classname>(PyObject *obj, void *ptr)`` - functions that can
  be used in ``PyArg_ParseTuple`` for converting an argument (given as
  ``PyObject *`` to an instance of ``<classname>``. Nothing needs to be
  programmed for the conversion, it is just a
  cast: ``*(GCPtr< T##type > *)(ptr) = PyOrange_As##type(obj);``). The
  difference between ``cc`` and ``ccn`` is that the latter accepts null
  pointers.

* ``TOrangeType`` that (essentially) inherits ``PyTypeObject``. The new
  definition also includes the RTTI used for wrapping (this way Orange knows
  which C++ class corresponds to which Python class), a pointer to the default
  constructor (used by generic constructors), a pointer to list of constructor
  keywords (``CONSTRUCTOR_KEYWORDS``, keyword arguments that should be ignored
  in a later call to ``init``) and recognized attributes
  (``RECOGNIZED_ATTRIBUTES``, attributes that don't yield warnings when set), a
  list of aliases, and pointers to ``cc_`` and ``ccn_`` functions. The latter
  are not used by Orange, since it can call the converters directly. They are
  here because ``TOrangeType`` is exported in a DLL while ``cc_`` and ``ccn_``
  are not (for the sake of limiting the number of exported symbols).


initialization.px
-----------------

Initialization.px defines the global module stuff.

First, here is a list of all ``TOrangeTypes``. The list is used for checking
whether some Python object is of Orange's type or derived from one, for finding
a Python class corresponding to a C++ class (based on C++'s RTTI). Orange also
exports the list as ``orange._orangeClasses``; this is a ``PyCObject`` so it can
only be used by other Python extensions written in C.

Then come declarations of all non-member functions, followed by a
``PyMethodDef`` structure with them.

Finally, here are declarations of functions that return manually constructed
constants (eg ``VarTypes``) and declarations of functions that add class
constants (eg ``Classifier_addConstants``). The latter functions were generated
by pyxtract and reside in the individual px files. Then follows a function that
calls all the constant related functions declared above. This function also adds
all class types to the Orange module.

The main module now only needs to call ``addConstants``.

externs.px
----------

Externs.px declares symbols for all Orange classes, for instance ::

    extern ORANGE_API TOrangeType PyOrDomain_Type;
    #define PyOrDomain_Check(op) PyObject_TypeCheck(op, (PyTypeObject *)&PyOrDomain_Type)
    int cc_Domain(PyObject *, void *);
    int ccn_Domain(PyObject *, void *);
    #define PyOrange_AsDomain(op) (GCPtr< TDomain >(PyOrange_AS_Orange(op)))

**************************
What and where to include?
**************************

As already mentioned, ppp files should be included (at the beginning) of the
corresponding cpp files, instead of the hpp file. For instance, domain.ppp is
included in domain.cpp. Each ppp should be compiled only once, all other files
needing the definition of ``TDomain`` should still include domain.hpp as usual.

File-specific px files are included in the corresponding cpp files.
lib_kernel.px is included at the end of lib_kernel.cpp, from which it was
generated. initialization.px should preferably be included in the file that
initializes the module (function ``initorange`` needs to call ``addConstants``,
which is declared in initialization.px. These px files contain definitions and
must be compiled only once. externs.px contains declarations and can be included
wherever needed.

For Microsoft Visual Studio, create a new, blank workspace. Specify the
directory with orange sources as "Location". Add a new project of type "Win 32
Dynamic-Link Library"; change the
location back to d:\ai\orange\source. Make it an empty DLL project.

Whatever names you give your module, make sure that the .cpp and .hpp files you
create as you go on are in orange\source\something (replace "something" with
something), since the further instructions will suppose it.
