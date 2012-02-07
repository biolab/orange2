.. py:currentmodule:: Orange.data

===============================
Domain description (``Domain``)
===============================

In Orange, the term `domain` denotes a set of variables and meta
attributes that describe data. A domain descriptor is attached to data
instances, data tables, classifiers and other objects. A descriptor is
constructed, for instance, after reading data from a file.

    >>> data = Orange.data.Table("zoo")
    >>> domain = data.domain
    >>> domain
    [hair, feathers, eggs, milk, airborne, aquatic, predator, toothed,
    backbone, breathes, venomous, fins, legs, tail, domestic, catsize,
    type], {-2:name}

Domains consists of ordinary features (from "hair" to "catsize" in the
above example), the class attribute ("type"), and meta attributes
("name"). We will refer to features and the class attribute as
*variables*. Variables are printed out in a form similar to a list whose
elements are attribute names, and meta attributes are printed like a
dictionary whose "keys" are meta attribute id's and "values" are
attribute names. In the above case, each data instance corresponds to an
animal and is described by the animal's properties and its type (the
class); the meta attribute contains the animal's name.

Domains as lists and dictionaries
=================================

Domains behave like lists: the length of domain is the number of
variables including the class variable. Domains can be indexed by integer
indices, variable names or instances of
:obj:`Orange.feature.Descriptor`::

    >>> domain["feathers"]
    EnumVariable 'feathers'
    >>> domain[1]
    EnumVariable 'feathers'
    >>> feathers = domain[1]
    >>> domain[feathers]
    EnumVariable 'feathers'

Meta attributes work the same::

    >>> domain[-2]
    StringVariable 'name'
    >>> domain["name"]
    StringVariable 'name'


Slices can be retrieved, but not set. Iterating through domain goes
through features and the class variable, but not through meta attributes::

    >>> for attr in domain:
    ...     print attr.name,
    ...
    hair feathers eggs milk airborne aquatic predator toothed backbone
    breathes venomous fins legs tail domestic catsize type

Method :obj:`Domain.index` returns the index of a variable specified by a
descriptor or name::

    >>> domain.index("feathers")
    1
    >>> domain.index(feathers)
    1
    >>> domain.index("name")
    -2


Conversions between domains
===========================

Domain descriptors can convert instances from one domain to another
(details on construction of domains are described later). ::

     >>> new_domain = Orange.data.Domain(["feathers", "legs", "type"], domain)
     >>> inst = data[55]
     >>> inst
     ['1', '0', '0', '1', '0', '0', '0', '1', '1', '1', '0', '0', '4',
     '1', '0', '1', 'mammal'], {"name":'oryx'}
     >>> inst2 = new_domain(inst)
     >>> inst2
     ['0', '4', 'mammal']

This is used, for instance, in classifiers: classifiers are often
trained on a preprocessed domain (e.g. on a subset of features or
on discretized data) and later used on instances from the original
domain. Classifiers store the training domain descriptor and use it
for converting new instances.

Alternatively, instances can be converted by constructing a new instance
and pass the new domain to the constructor::

     >>> inst2 = Orange.data.Instance(new_domain, inst)

Entire data table can be converted in a similar way::

     >>> data2 = Orange.data.Table(new_domain, data)
     >>> data2[55]
     ['0', '4', 'mammal']


.. _multiple-classes:

Multiple classes
================

A domain can have multiple additional class attributes. These are stored
similarly to other features except that they are not used for learning. The
list of such classes is stored in :obj:`~Orange.data.Domain.class_vars`.
When converting between domains, multiple classes can become ordinary
features or the class, and vice versa.

.. _meta-attributes:

Meta attributes
===============

Meta attributes hold additional data attached to individual
instances. Different instances from the same domain or even the same
table may have different meta attributes. (See documentation on
:obj:`Orange.data.Instance` for details about meta values.)

Meta attributes that appear in instances can - but don't need to - be
listed in the domain. Typically, the meta attribute will be included in
the domain for the following reasons.

     * If the domain knows about meta attributes, their values can be
       obtained with indexing by names and variable descriptors,
       e.g. ``inst["age"]``. Values of unknown meta attributes
       can be obtained only through integer indices (e.g. inst[id], where
       id needs to be an integer).

     * When printing out a data instance, the symbolic values of discrete
       meta attributes can only be printed if the attribute is
       registered. Also, if the attribute is registered, the printed
       out example will show a (more informative) attribute's name
       instead of a meta-id.

     * When saving instances to a file, only the values of registered
       meta attributes are saved.

     * When a new data instance is constructed, it will have all the
       meta attributes listed in the domain, with their values set to
       unknown.

Meta attribute can be marked as "optional". Non-optional meta
attributes are *expected to be* present in all data instances from that
domain. This rule is not strictly enforced. As one of the few places
where the difference matters, saving to files fails if a non-optional
meta value is missing; optional attributes are not written to the file
at all. Also, newly constructed data instances initially have all the
non-optional meta attributes.

While the list of features and the class value are immutable,
meta attributes can be added and removed at any time::

     >>> misses = Orange.feature.Continuous("misses")
     >>> id = Orange.feature.Descriptor.new_meta_id()
     >>> data.domain.add_meta(id, misses)

This does not change the data: no attributes are added to data
instances. Methods related to meta attributes are described in more
details later.

Registering meta attributes enables addressing by indexing, either by
name or by descriptor. For instance, the following snippet sets the new
attribute to 0 for all instances in the data table::

     >>> for inst in data:
     ...     inst[misses] = 0

An alternative is to refer to the attribute by name::

     >>> for inst in data:
     ...     inst["misses"] = 0

If the attribute were not registered, it could still be set using the
integer index::

     >>> for inst in data:
     ...    inst.set_meta(id, 0)

Registering the meta attribute also enhances printouts. When an instance
is printed, meta-values for registered meta attributes are shown as
"name:value" pairs, while for unregistered only id is given instead
of a name.

A meta-attribute can be used, for instance, to record the number of
misclassifications by a given ``classifier``::

     >>> for inst in data:
     ... if inst.get_class() != classifier(inst):
     ...     inst[misses] += 1

The other effect of registering meta attributes is that they appear in
converted instances: whenever an instances is converted to some
domain, it will have all the meta attributes that are registered in
that domain. If the meta attributes occur in the original domain of
the instance or if they can be computed from them, they will have
appropriate values, otherwise their value will be missing. ::

    new_domain = Orange.data.Domain(["feathers", "legs"], domain)
    new_domain.add_meta(Orange.feature.Descriptor.new_meta_id(), domain["type"])
    new_domain.add_meta(Orange.feature.Descriptor.new_meta_id(), domain["legs"])
    new_domain.add_meta(
        Orange.feature.Descriptor.new_meta_id(), Orange.feature.Discrete("X"))
    data2 = Orange.data.Table(new_domain, data)

Domain ``new_domain`` in this example has variables ``feathers`` and
``legs`` and meta attributes ``type``, ``legs`` (again) and ``X`` which
is a new feature with no relation to the existing ones. ::

    >>> data[55]
    ['1', '0', '0', '1', '0', '0', '0', '1', '1', '1', '0', '0',
    '4', '1', '0', '1', 'mammal'], {"name":'oryx'}
    >>> data2[55]
    ['0', '4'], {"type":'mammal', "legs":'4', "X":'?'}



.. class:: Domain

     .. attribute:: features

         Immutable list of domain attributes without the class
         variable. Read only.

     .. attribute:: variables

         List of domain attributes including the class variable. Read only.

     .. attribute:: class_var

         The class variable (:obj:`~Orange.feature.Descriptor`) or
         ``None``. Read only.

     .. attribute:: class_vars

         A list of additional class attributes. Read only.

     .. attribute:: version

         An integer value that is changed when the domain is
         modified. The value can be also used as unique domain identifier; two
         different domains have different value of ``version``.

     .. method:: __init__(variables[, class_vars=])

         Construct a domain with the given variables; the
         last one is used as the class variable. ::

             >>> a, b, c = [Orange.feature.Discrete(x) for x in "abc"]
             >>> domain = Orange.data.Domain([a, b, c])
             >>> domain.features
             <EnumVariable 'a', EnumVariable 'b'>
             >>> domain.class_var
             EnumVariable 'c'

         :param variables: List of variables (instances of :obj:`~Orange.feature.Descriptor`)
         :type variables: list
         :param class_vars: A list of multiple classes; must be a keword argument
         :type class_vars: list

     .. method:: __init__(features, class_variable[, class_vars=])

         Construct a domain with the given list of features and the
         class variable. ::

             >>> domain = Orange.data.Domain([a, b], c)
             >>> domain.features
             <EnumVariable 'a', EnumVariable 'b'>
             >>> domain.class_var
             EnumVariable 'c'

         :param features: List of features (instances of :obj:`~Orange.feature.Descriptor`)
         :type features: list
         :param class_variable: Class variable
         :type class_variable: Orange.feature.Descriptor
         :param class_vars: A list of multiple classes; must be a keyword argument
         :type class_vars: list

     .. method:: __init__(variables, has_class[, class_vars=])

         Construct a domain with the given variables. If ``has_class``
         is ``True``, the last variable is the class. ::

             >>> domain = Orange.data.Domain([a, b, c], False)
             >>> domain.features
             <EnumVariable 'a', EnumVariable 'b'>
             >>> domain.class_var
             EnumVariable 'c'

         :param variables: List of variables (instances of :obj:`~Orange.feature.Descriptor`)
         :type features: list
         :param has_class: A flag telling whether the domain has a class
         :type has_class: bool
         :param class_vars: A list of multiple classes; must be a keyword argument
         :type class_vars: list

     .. method:: __init__(variables, source[, class_vars=])

         Construct a domain with the given variables. Variables specified
         by names are sought for in the ``source`` argument. The last
         variable from the list is used as the class variable. ::

             >>> domain1 = Orange.data.Domain([a, b])
             >>> domain2 = Orange.data.Domain(["a", b, c], domain)

         :param variables: List of variables (strings or instances of :obj:`~Orange.feature.Descriptor`)
         :type variables: list
         :param source: An existing domain or a list of variables
         :type source: Orange.data.Domain or list of :obj:`~Orange.feature.Descriptor`
         :param class_vars: A list of multiple classes; must be a keyword argument
         :type class_vars: list

     .. method:: __init__(variables, has_class, source[, class_vars=])

         Similar to above except for the flag which tells whether the
         last variable should be used as the class variable. ::

             >>> domain1 = Orange.data.Domain([a, b], False)
             >>> domain2 = Orange.data.Domain(["a", b, c], False, domain)

         :param variables: List of variables (strings or instances of :obj:`~Orange.feature.Descriptor`)
         :type variables: list
         :param has_class: A flag telling whether the domain has a class
         :type has_class: bool
         :param source: An existing domain or a list of variables
         :type source: Orange.data.Domain or list of :obj:`~Orange.feature.Descriptor`
         :param class_vars: A list of multiple classes; must be a keyword argument
         :type class_vars: list

     .. method:: __init__(domain, class_var[, class_vars=])

         Construct a copy of an existing domain except that the class
         variable is replaced with the one specified in the argument
         and the class variable of the existing domain becomes an
         ordinary feature. If the new class is one of the original
         domain's features, ``class_var`` can also be specified by name.

         :param domain: An existing domain
         :type domain: :obj:`~Orange.variable.Domain`
         :param class_var: Class variable for the new domain
         :type class_var: :obj:`~Orange.feature.Descriptor` or string
         :param class_vars: A list of multiple classes; must be a keyword argument
         :type class_vars: list

     .. method:: __init__(domain, has_class=False[, class_vars=])

         Construct a copy of the domain. If the ``has_class``
         flag is given and is :obj:`False`, the class attribute becomes
         an ordinary feature.

         :param domain: An existing domain
         :type domain: :obj:`~Orange.variable.Domain`
         :param has_class: A flag indicating whether the domain will have a class
         :type has_class: bool
         :param class_vars: A list of multiple classes; must be a keword argument
         :type class_vars: list

     .. method:: has_discrete_attributes(include_class=True)

         Return ``True`` if the domain has any discrete variables;
         class is included unless ``include_class`` is ``False``.

         :param has_class: tells whether to consider the class variable
         :type has_class: bool
         :rtype: bool

     .. method:: has_continuous_attributes(include_class=True)

         Return ``True`` if the domain has any continuous variables;
         class is included unless ``include_class`` is ``False``.

         :param has_class: tells whether to consider the class variable
         :type has_class: bool
         :rtype: bool

     .. method:: has_other_attributes(include_class=True)

         Return ``True`` if the domain has any variables that are
         neither discrete nor continuous, such as, for instance string
         variables. The class is included unless ``include_class`` is
         ``False``.

         :param has_class: tells whether to consider the class variable
         :type has_class: bool
         :rtype: bool


     .. method:: add_meta(id, variable, optional=0)

         Register a meta attribute with the given id (see
         :obj:`Orange.feature.Descriptor.new_meta_id`). The same meta attribute should
         have the same id in all domains in which it is registered. ::

             >>> newid = Orange.feature.Descriptor.new_meta_id()
             >>> domain.add_meta(newid, Orange.feature.String("origin"))
             >>> data[55]["origin"] = "Nepal"
             >>> data[55]
             ['1', '0', '0', '1', '0', '0', '0', '1', '1', '1', '0', '0',
             '4', '1', '0', '1', 'mammal'], {"name":'oryx', "origin":'Nepal'}

         The third argument tells whether the meta attribute is optional or
         not; non-zero values indicate optional attributes. Different
         values can be used to distinguish between various types
         optional attributes; the meaning of the value is not defined in
         advance and can be used arbitrarily by the application.

         :param id: id of the new meta attribute
         :type id: int
         :param variable: variable descriptor
         :type variable: Orange.feature.Descriptor
         :param optional: indicates whether the meta attribute is optional
         :type optional: int

     .. method:: add_metas(attributes, optional=0)

         Add multiple meta attributes at once. The dictionary contains id's as
         keys and variables (:obj:`~Orange.feature.Descriptor`) as the
         corresponding values. The following example shows how to add all
         meta attributes from another domain::

              >>> newdomain.add_metas(domain.get_metas())

         The optional second argument has the same meaning as in :obj:`add_meta`.

         :param attributes: dictionary of id's and variables
         :type attributes: dict
         :param optional: tells whether the meta attribute is optional
         :type optional: int

     .. method:: remove_meta(attribute)

         Removes one or multiple meta attributes. Removing a meta attribute has
         no effect on data instances.

         :param attribute: attribute(s) to be removed, given as name, id, variable descriptor or a list of them
         :type attribute: string, int, Orange.feature.Descriptor; or a list

     .. method:: has_attribute(attribute)

         Return ``True`` if the domain contains the specified meta
         attribute.

         :param attribute: attribute to be checked
         :type attribute: string, int, Orange.feature.Descriptor
         :rtype: bool

     .. method:: meta_id(attribute)

         Return an id of a meta attribute.

         :param attribute: name or variable descriptor of the attribute
         :type attribute: string or Orange.feature.Descriptor
         :rtype: int

     .. method:: get_meta(attribute)

         Return a variable descriptor corresponding to the meta attribute.

         :param attribute: name or id of the attribute
         :type attribute: string or int
         :rtype: Orange.feature.Descriptor

     .. method:: get_metas()

          Return a dictionary with meta attribute id's as keys and
          corresponding variable descriptors as values.

     .. method:: get_metas(optional)

          Return a dictionary with meta attribute id's as keys and
          corresponding variable descriptors as values. The dictionary
          contains only meta attributes for which the argument ``optional``
          matches the flag given when the attributes were added using
          :obj:`add_meta` or :obj:`add_metas`.

          :param optional: flag that specifies the attributes to be returned
          :type optional: int
          :rtype: dict

     .. method:: is_optional_meta(attribute)

         Return ``True`` if the given meta attribute is optional,
         and ``False`` if it is not.

         :param attribute: attribute to be checked
         :type attribute: string, int, Orange.feature.Descriptor
         :rtype: bool
