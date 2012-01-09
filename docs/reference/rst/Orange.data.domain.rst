.. py:currentmodule:: Orange.data

===============================
Domain description (``Domain``)
===============================

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
variables, including the class variable. Iterating through domain
goes through features and the class variable, but not through meta
attributes. Domains can be indexed by integer indices, variable names
or instances of :obj:`Orange.data.variables.Variable`. Domain has a
method :obj:`Domain.index` that returns the index of a variable
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

Conversions between domains
===========================

Domain descriptors are used to convert instances from one domain to
another. ::

     >>> data = Orange.data.Table("monk1")
     >>> d2 = Orange.data.Domain(["a", "b", "e", "y"], data.domain)
     >>> 
     >>> inst = data[55]
     >>> print inst
     ['1', '2', '1', '1', '4', '2', '0']
     >>> inst2 = d2(inst)
     >>>  print inst2
     ['1', '2', '4', '0']

This is used, for instance, in classifiers: classifiers are often
trained on a preprocessed domain (e.g. with a subset of features or
with discretized data) and later used on instances from the original
domain. Classifiers store the training domain descriptor and use it
for converting new instances.

Equivalently, instances can be converted by passing the new domain to
the constructor::

     >>> inst2 = Orange.data.Instance(d2, inst)

Entire data table can be converted similarly::

     >>> data2 = Orange.data.Table(d2, data)
     >>> print data2[55]
     ['1', '2', '4', '0']


Multiple classes
================

A domain can have multiple additional class attributes. These are stored
similarly to other features except that they are not used for learning. The
list of such classes is stored in `class_vars`. When converting between
domains, multiple classes can become ordinary features or the class, and
vice versa.

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
     >>> id = Orange.data.new_meta_id()
     >>> data.domain.add_meta(id, misses)

This does not change the data: no attributes are added to data
instances.

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
     ... if inst.get_class() != classifier(example):
     ...     example[misses] += 1

The other effect of registering meta attributes is that they appear in
converted instances: whenever an instances is converted to some
domain, it will have all the meta attributes that are registered in
that domain. If the meta attributes occur in the original domain of
the instance or if they can be computed from them, they will have
appropriate values, otherwise they will have a "don't know" value. ::

     domain = data.domain
     d2 = Orange.data.Domain(["a", "b", "e", "y"], domain)
     for attr in ["c", "d", "f"]:
	 d2.add_meta(Orange.data.new_meta_id(), domain[attr])
     d2.add_meta(Orange.data.new_meta_id(), orange.data.variable.Discrete("X"))
     data2 = Orange.data.Table(d2, data)

Domain ``d2`` in this example has variables ``a``, ``b``, ``e`` and the
class, while the other three variables are added as meta
attributes, together with additional attribute X. Results are as
follows. ::

     >>> print data[55]
     ['1', '2', '1', '1', '4', '2', '0'], {"misses":0.000000}
     >>> print data2[55]
     ['1', '2', '4', '0'], {"c":'1', "d":'1', "f":'2', "X":'?'}

After conversion, the three attributes are moved to meta attributes
and the new attribute appears as unknown.



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

     .. attribute:: class_vars

	 A list of additional class attributes. Read only.

     .. attribute:: version

	 An integer value that is changed when the domain is
	 modified. Can be also used as unique domain identifier; two
	 different domains also have different versions.

     .. method:: __init__(variables[, class_vars=])

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
         :param class_vars: A list of multiple classes; must be a keword argument
	 :type variables: list

     .. method:: __init__(features, class_variable[, classVars=])

	 Construct a domain with the given list of features and the
	 class variable. ::

	     >>> d = Orange.data.Domain([a, b], c)
	     >>> print d.features
	     <EnumVariable 'a', EnumVariable 'b'>
	     >>> print d.class_var EnumVariable 'c'

	 :param features: List of features (instances of :obj:`Orange.data.variable.Variable`)
	 :type features: list
	 :param class_variable: Class variable
         :param class_vars: A list of multiple classes; must be a keword argument
	 :type features: Orange.data.variable.Variable

     .. method:: __init__(variables, has_class[, class_vars=])

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
         :param class_vars: A list of multiple classes; must be a keword argument
	 :type has_class: bool

     .. method:: __init__(variables, source[, class_vars=])

	 Construct a domain with the given variables, which can also be
	 specified by names, provided that the variables with that
	 names exist in the source list. The last variable from the
	 list is used as the class variable. ::

	     >>> d1 = orange.Domain([a, b])
	     >>> d2 = orange.Domain(["a", b, c], d1) 

	 :param variables: List of variables (strings or instances of :obj:`Orange.data.variable.Variable`)
	 :type variables: list
	 :param source: An existing domain or a list of variables
         :param class_vars: A list of multiple classes; must be a keword argument
	 :type source: Orange.data.Domain or list of :obj:`Orange.data.variable.Variable`

     .. method:: __init__(variables, has_class, source[, class_vars=])

	 Similar to above except for the flag which tells whether the
	 last variable should be used as the class variable. ::

	     >>> d1 = orange.Domain([a, b])
	     >>> d2 = orange.Domain(["a", b, c], d1) 

	 :param variables: List of variables (strings or instances of :obj:`Orange.data.variable.Variable`)
	 :type variables: list
	 :param has_class: A flag telling whether the domain has a class
	 :type has_class: bool
	 :param source: An existing domain or a list of variables
         :param class_vars: A list of multiple classes; must be a keword argument
	 :type source: Orange.data.Domain or list of :obj:`Orange.data.variable.Variable`

     .. method:: __init__(domain, class_var[, class_vars=])

	 Construct a domain as a shallow copy of an existing domain
	 except that the class variable is replaced with the given one
	 and the class variable of the existing domain becoems an
	 ordinary feature. If the new class is one of the original
	 domain's features, it can also be specified by a name.

	 :param domain: An existing domain
	 :type domain: :obj:`Orange.variable.Domain`
	 :param class_var: Class variable for the new domain
         :param class_vars: A list of multiple classes; must be a keword argument
	 :type class_var: string or :obj:`Orange.data.variable.Variable`

     .. method:: __init__(domain, has_class=False[, class_vars=])

	 Construct a shallow copy of the domain. If the ``has_class``
	 flag is given and equals :obj:`False`, it moves the class
	 attribute to ordinary features.

	 :param domain: An existing domain
	 :type domain: :obj:`Orange.variable.Domain`
	 :param has_class: A flag telling whether the domain has a class
         :param class_vars: A list of multiple classes; must be a keword argument
	 :type has_class: bool

     .. method:: has_discrete_attributes(include_class=True)

	 Return :obj:`True` if the domain has any discrete variables;
	 class is considered unless ``include_class`` is ``False``.

	 :param has_class: Tells whether to consider the class variable
	 :type has_class: bool
	 :rtype: bool

     .. method:: has_continuous_attributes(include_class=True)

	 Return :obj:`True` if the domain has any continuous variables;
	 class is considered unless ``include_class`` is ``False``.

	 :param has_class: Tells whether to consider the class variable
	 :type has_class: bool
	 :rtype: bool

     .. method:: has_other_attributes(include_class=True)

	 Return :obj:`True` if the domain has any variables which are
	 neither discrete nor continuous, such as, for instance string variables.
	 class is considered unless ``include_class`` is ``False``.

	 :param has_class: Tells whether to consider the class variable
	 :type has_class: bool
	 :rtype: bool


     .. method:: add_meta(id, variable, optional=0)

	 Register a meta attribute with the given id (obtained by
	 :obj:`Orange.data.new_meta_id`). The same meta attribute can (and
	 should) have the same id when registered in different domains. ::

	     >>> newid = Orange.data.new_meta_id()
	     >>> d2.add_meta(newid, Orange.data.variable.String("name"))
	     >>> d2[55]["name"] = "Joe"
	     >>> print data2[55]
	     ['1', '2', '4', '0'], {"c":'1', "d":'1', "f":'2', "X":'?', "name":'Joe'}

	 The third argument tells whether the meta attribute is optional or
	 not. The parameter is an integer, with any non-zero value meaning that
	 the attribute is optional. Different values can be used to distinguish
	 between various optional attributes; the meaning of the value is not
	 defined in advance and can be used arbitrarily by the application.

	 :param id: id of the new meta attribute
	 :type id: int
	 :param variable: variable descriptor
	 :type variable: Orange.data.variable.Variable
	 :param optional: tells whether the meta attribute is optional
	 :type optional: int

     .. method:: add_metas(attributes, optional=0)

	 Add multiple meta attributes at once. The dictionary contains id's as
	 keys and variables as the corresponding values. The following example
	 shows how to add all meta attributes from one domain to another::

	      newdomain.add_metas(domain.get_metas)

	 The optional second argument has the same meaning as in :obj:`add_meta`.

	 :param attributes: dictionary of id's and variables
	 :type attributes: dict
	 :param optional: tells whether the meta attribute is optional
	 :type optional: int

     .. method:: remove_meta(attribute)

	 Removes one or multiple meta attributes. Removing a meta attribute has
	 no effect on data instances.

	 :param attribute: attribute(s) to be removed, given as name, id, variable descriptor or a list of them
	 :type attribute: string, int, Orange.data.variable.Variable; or a list

     .. method:: has_attribute(attribute)

	 Return True if the domain contains the specified meta attribute.

	 :param attribute: attribute to be checked
	 :type attribute: string, int, Orange.data.variable.Variable
	 :rtype: bool

     .. method:: meta_id(attribute)

	 Return an id of a meta attribute.

	 :param attribute: name or variable descriptor of the attribute
	 :type attribute: string or Orange.data.variable.Variable
	 :rtype: int

     .. method:: get_meta(attribute)

	 Return a variable descriptor corresponding to the meta attribute.

	 :param attribute: name or id of the attribute
	 :type attribute: string or int
	 :rtype: Orange.data.variable.Variable

     .. method:: get_metas()

	  Return a dictionary with meta attribute id's as keys and corresponding
	  variable descriptors as values.

     .. method:: get_metas(optional)

	  Return a dictionary with meta attribute id's as keys and corresponding
	  variable descriptors as values; the dictionary contains only meta
	  attributes for which the argument ``optional`` matches the flag given
	  when the attributes were added using :obj:`add_meta` or :obj:`add_metas`.

	  :param optional: flag that specifies the attributes to be returned
	  :type optional: int
	  :rtype: dict

     .. method:: is_optional_meta(attribute)

	 Return True if the given meta attribute is optional, and False if it is
	 not.

	 :param attribute: attribute to be checked
	 :type attribute: string, int, Orange.data.variable.Variable
	 :rtype: bool
