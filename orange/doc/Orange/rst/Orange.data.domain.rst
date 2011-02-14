==================
Domain description
==================

In Orange, the term `domain` denotes a set of features, which will be
used to describe the data instances, the class variables, meta
attributes and similar. Each data instance, as well as many
classifiers and other objects are associated with a domain descriptor,
which defines the object's content and/or its input and output data
format.

Domain descriptors are also responsible for converting data instances
from one domain to another, e.g. from the original feature space to
one with different set of features which are selected or constructed
from the original set.

Domains as lists
================

Domains resemble lists: the length of domain is the number of
variables, including the class attribute. Iterating through domain
goes through attributes and the class attribute, but not through meta
attributes. Domains can be indexed by integer indices, variable names
or instances of :obj:`Orange.data.variables.Variable`. Domain has a
method :obj:`Domain.index` that returns the index of an attribute
specified by a descriptor, name. Slices can be retrieved, but not
set. ::

    >>> print d2
    [a, b, e, y], {-4:c, -5:d, -6:f, -7:X}
    >>> d2[1]
    EnumVariable 'b'
    >>> d2["e"]
    EnumVariable 'e'
    >>> d2["d"]
    EnumVariable 'd'
    >>> d2[-4]
    EnumVariable 'c'
    >>> for attr in d2:
    ...     print attr.name,
    ...
    a b e y 

Meta attributes
===============

Meta-values are additional values that can be attached to instances.
It is not necessary that all instances in the same table (or even all
instances from the same domain) have certain meta-value. See documentation
on :obj:`Orange.data.Instance` for a more thorough description of meta-values.

Meta attributes that appear in instances can, but don't need to be
registered in the domain. Typically, the meta attribute will be
registered for the following reasons.

    * If the domain knows about a meta attribute, their values can be
      obtained with indexing by names and variable descriptors,
      e.g. ``inst["age"]``. Values of unregistered meta attributes can
      be obtained only through integer indices (e.g. inst[id], where
      id needs to be an integer).

    * When printing out an instance, the symbolic values of discrete
      meta attributes can only be printed if the attribute is
      registered. Also, if the attribute is registered, the printed
      out example will show a (more informative) attribute's name
      instead of a meta-id.

    * Registering an attribute provides a way to attach a descriptor
      to a meta-id. See how the basket file format uses this feature.

    * When saving instances to a file, only the values of registered
      meta attributes are saved.

    * When a new data instance is constructed, it is automatically
      assigned the meta attributes listed in the domain, with their
      values set to unknown.

For the latter two points - saving to a file and construction of new
instances - there is an additional flag: a meta attribute can be
marked as "optional". Such meta attributes are not saved and not added
to newly constructed data instances. This functionality is used in,
for instance, the above mentioned basket format, where new meta
attributes are created while loading the file and new instances to
contain all words from the past examples.

There is another distinction between the optional and non-optional
meta attributes: the latter are `expected to be` present in all
examples of that domain. Saving to files expects them and will fail if
a non-optional meta value is missing. Optional attributes may be
missing. In most other places, these rules are not strictly enforced,
so adhering to them is rather up to choice.

Meta attributes can be added and removed even after the domain is
constructed and instances of that domain already exist. For instance,
if data contains the Monk 1 data set, we can add a new continuous
attribute named "misses" with the following code (a detailed
desription of methods related to meta attributes is given below)::

    >>> misses = Orange.data.variable.Continuous("misses")
    >>> id = orange.newmetaid()
    >>> data.domain.addmeta(id, misses)

This does not change the data: no attributes are added to data
instances.

Registering meta attributes enables addressing by indexing, either by
name or by descriptor. For instance, the following snippet sets the new
attribute to 0 for all instances in the data table::

    >>> for inst in data:
    ...     inst[misses] = 0

An alternative is referring by name::

    >>> for inst in data:
    ...     inst["misses"] = 0

If the attribute were not registered, it could still be set using the
integer index::

    >>> for inst in data:
    ...    inst.setmeta(id, 0)

Registering the meta attribute also enhances printouts. When an instance
is printed, meta-values for registered meta attributes are shown as
"name:value" pairs, while for unregistered only id is given instead
of a name.

In a massive testing of different models, you could count the number
of times that each example was missclassified by calling classifiers
in the following loop.

domain2.py (uses monk1.tab) &gt;&gt;&gt; for example in data: ... if
example.getclass() != classifier(example): ... example[misses] += 1

The other effect of registering meta attributes is that they appear in converted examples. That is, whenever an example is converted to certain domain, the example will have all the meta attributes that are declared in that domain. If the meta attributes occur in the original domain of the example or can be computed from the attributes in the original domain, they will have appropriate values. When not, their values will be DK.
domain = data.domain d2 = orange.Domain(["a", "b", "e", "y"], domain) for attr in ["c", "d", "f"]: d2.addmeta(orange.newmetaid(), domain[attr]) d2.addmeta(orange.newmetaid(), orange.EnumVariable("X")) data2 = orange.ExampleTable(d2, data)

Domain d2 is constructed to have only the attributes a, b, e and the class attribute, while the other three attributes are added as meta attributes, among with a mysterious additional attribute X.
&gt;&gt;&gt; print data[55] ['1', '2', '1', '1', '4', '2', '0'], {"misses":0.000000} &gt;&gt;&gt; print data2[55] ['1', '2', '4', '0'], {"c":'1', "d":'1', "f":'2', "X":'?'}

After conversion, the three attributes are moved to meta attributes and the new attribute appears as unknown.

.. class:: Domain

    .. attribute:: features

        List of domain attributes
        (:obj:`Orange.data.variable.Variables`) without the class
        variable. Read only.

    .. attribute:: variables

        List of domain attributes
        (:obj:`Orange.data.variable.Variables`) including the class
        variable. Read only.

    .. attribute:: class_var

        The class variable (:obj:`Orange.data.variable.Variable`), or
        :obj:`None` if there is none. Read only.

    .. attribute:: version

        An integer value that is changed when the domain is
        modified. Can be also used as unique domain identifier; two
        different domains also have different versions.

    .. method:: __init__(variables)

        Construct a domain with the given variables specified; the
        last one is used as the class variable. ::

            >>> a, b, c = [Orange.data.variable.Discrete(x)
                           for x in ["a", "b", "c"]]
            >>> d = Orange.data.Domain([a, b, c])
            >>> print d.features
            <EnumVariable 'a', EnumVariable 'b'>
            >>> print d.class_var
            EnumVariable 'c'

        :param variables: List of variables (instances of :obj:`Orange.data.variable.Variable`)
        :type variables: list

    .. method:: __init__(features, class_variable)

        Construct a domain with the given list of features and the
        class variable. ::

            >>> d = Orange.data.Domain([a, b], c)
            >>> print d.features
            <EnumVariable 'a', EnumVariable 'b'>
            >>> print d.class_var EnumVariable 'c'

        :param features: List of features (instances of :obj:`Orange.data.variable.Variable`)
        :type features: list
        :param class_variable: Class variable
        :type features: Orange.data.variable.Variable

    .. method:: __init__(variables, has_class)

        Construct a domain with the given variables. If has_class is
        :obj:`True`, the last one is used as the class variable. ::

            >>> d = Orange.data.Domain([a, b, c], False)
            >>> print d.features
            <EnumVariable 'a', EnumVariable 'b'>
            >>> print d.class_var
            EnumVariable 'c'

        :param variables: List of variables (instances of :obj:`Orange.data.variable.Variable`)
        :type features: list
        :param has_class: A flag telling whether the domain has a class
        :type has_class: bool

    .. method:: __init__(variables, source)

        Construct a domain with the given variables, which can also be
        specified by names, provided that the variables with that
        names exist in the source list. The last variable from the
        list is used as the class variable. ::

            >>> d1 = orange.Domain([a, b])
            >>> d2 = orange.Domain(["a", b, c], d1) 

        :param variables: List of variables (strings or instances of :obj:`Orange.data.variable.Variable`)
        :type variables: list
        :param source : An existing domain or a list of variables
        :type source: Orange.data.Domain or list of
        :obj:`Orange.data.variable.Variable`

    .. method:: __init__(variables, has_class, source)

        Similar to above except for the flag which tells whether the
        last variable should be used as the class variable. ::

            >>> d1 = orange.Domain([a, b])
            >>> d2 = orange.Domain(["a", b, c], d1) 

        :param variables: List of variables (strings or instances of :obj:`Orange.data.variable.Variable`)
        :type variables: list
        :param has_class: A flag telling whether the domain has a class
        :type has_class: bool
        :param source : An existing domain or a list of variables
        :type source: Orange.data.Domain or list of
        :obj:`Orange.data.variable.Variable`

    .. method:: __init__(domain, class_var)

        Construct a domain as a shallow copy of an existing domain
        except that the class variable is replaced with the given one
        and the class variable of the existing domain becoems an
        ordinary feature. If the new class is one of the original
        domain's features, it can also be specified by a name.

        :param domain: An existing domain
        :type domain: :obj:`Orange.variable.Domain`
        :param class_var: Class variable for the new domain
        :type class_var: string or :obj:`Orange.data.variable.Variable`

    .. method:: __init__(domain[, has_class])

        Construct a shallow copy of the domain. If the ``has_class``
        flag is given and equals :obj:`False`, it moves the class
        attribute to ordinary features.

        :param domain: An existing domain
        :type domain: :obj:`Orange.variable.Domain`
        :param has_class: A flag telling whether the domain has a class
        :type has_class: bool

    .. method:: has_discrete_attributes([include_class=True])

        Return :obj:`True` if the domain has any discrete variables;
        class is considered unless ``include_class`` is ``False``.

        :param has_class: Tells whether to consider the class variable
        :type has_class: bool
        :rtype: bool

    .. method:: has_continuous_attributes([include_class=True])

        Return :obj:`True` if the domain has any continuous variables;
        class is considered unless ``include_class`` is ``False``.

        :param has_class: Tells whether to consider the class variable
        :type has_class: bool
        :rtype: bool

    .. method:: has_other_attributes([include_class=True])

        Return :obj:`True` if the domain has any variables which are
        neither discrete nor continuous, such as, for instance string variables.
        class is considered unless ``include_class`` is ``False``.

        :param has_class: Tells whether to consider the class variable
        :type has_class: bool
        :rtype: bool
