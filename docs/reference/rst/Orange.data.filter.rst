.. py:currentmodule:: Orange.data.filter

.. index:: filter

.. index::
   single: filtering; instance filtering

**********************
Filtering (``filter``)
**********************

Filters are used to select subsets of instances. Consider the following
example, where instances with age="young" from lenses are
selected:

.. literalinclude:: code/filter.py
    :lines: 58-64

Outputs::

    Young instances
    ['young', 'myope', 'no', 'reduced', 'none']
    ['young', 'myope', 'no', 'normal', 'soft']
    ['young', 'myope', 'yes', 'reduced', 'none']
    ['young', 'myope', 'yes', 'normal', 'hard']
    ['young', 'hypermetrope', 'no', 'reduced', 'none']
    ['young', 'hypermetrope', 'no', 'normal', 'soft']
    ['young', 'hypermetrope', 'yes', 'reduced', 'none']
    ['young', 'hypermetrope', 'yes', 'normal', 'hard']

:obj:`~Orange.data.Domain.features` behaves as a list and provides method
`index`, which is used to retrieve the position of feature `age`. Feature
`age` is also used to construct a :obj:`~Orange.data.Value`.

Structure
---------

Filters see individual instances, not the entire table,
and are limited to accepting or rejecting instances. All filters have this
structure:

.. class:: Filter

    .. attribute:: negate

    Inverts the selection. Defaults to :obj:`False`.

    .. attribute:: domain

    Domain to which examples are converted prior to checking.
    :obj:`Random` ignores this field.

    .. method:: __call__(instance)

    Checks whether the instance matches the filter's criterion and returns
    either :obj:`True` or :obj:`False`.

    .. method:: __call__(table)

    When given an entire data table, it returns a list of instances (as a
    :obj:`~Orange.data.Table`) that matches the criterion.

    .. method:: selectionVector(table)

    Returns a list of :obj:`bool` of the same length as :obj:`table`,
    denoting which instances are accepted. Equivalent to
    `[filter(ex) for i in table]`.

An alternative way to apply a filter is to call
:obj:`~Orange.data.Table.filter` on the data table.

Random filter
-------------

.. class:: Random

    It accepts an instance with a given probability.

    .. attribute:: prob

    Probability for accepting an instance.

    .. attribute:: random_generator

    The random number generator used for making selections. If not set
    before filtering, a new generator is constructed and stored here for
    later use.

.. literalinclude:: code/filter.py
    :lines: 12-14

The output is::

    1 0 0 0 1 1 0 1 0 1

In this script, :obj:`instance` should be some learning instance;
you can load any data and set `instance = data[0]`. Although the probability
of selecting an instance is set to 0.7, the filter accepted five out of ten
instances. Because the filter only sees individual instances, it cannot be
accurate in this regard. If exactly 70% of instances are needed then use
:obj:`~Orange.data.sample.SubsetIndices2`.

Setting the random generator ensures that the filter will always select
the same instances, disregarding of how many times you run the script or what
you do in Orange before you run it. Setting `randomGenerator=24` is a
shortcut for `randomGenerator = Orange.misc.Random(initseed=24)` or
`randomGenerator = Orange.misc.Random(initseed=24)`.

To select a subset of instances instead of calling the filter for each
individual example, use a filter like this::

    data70 = randomfilter(data)

Unknown values
--------------

.. class:: IsDefined

    This class selects instances for which all feature values are defined.
    By default, the filter checks all features; you can modify
    the list :obj:`check` to limit the features to check.
    This filter does not check meta attributes.

.. class:: HasSpecial

    This is an obsolete filter which selects instances with at least one
    unknown value in any feature.
    This filter does not check meta attributes.

.. class:: HasClass

    Selects instances with defined class value. You can use
    :obj:`~Orange.data.filter.Filter.negate` to invert the selection,
    as shown in the script below.

    .. attribute:: check

    A list of :obj:`bool` elements specifying which features to check. Each
    element corresponds to a feature in the domain. By default,
    :obj:`check` is :obj:`None`, meaning that all features are checked. The
    list is initialized to a list of :obj:`True` when the filter's
    :obj:`~Orange.data.filter.Filter.domain` is set unless the list
    already exists. You can also set
    :obj:`~Orange.data.filter.HasClass.check` manually,
    even without setting the :obj:`~Orange.data.filter.Filter.domain`. The list
    can be indexed by ordinary integers (for example,
    `check[0]`). If :obj:`~Orange.data.filter.Filter.domain` is set,
    you can also address the list by feature names or descriptors.

After setting :obj:`~Orange.data.filter.Filter.domain`
the :obj:`~Orange.data.Domain` should not be modified. Changes will
disrupt the correspondence between the domain features and the
list :obj:`~Orange.data.filter.HasClass.check`, causing unpredictable
behaviour.

.. literalinclude:: code/filter.py
    :lines: 9, 20-55

Meta values
-----------

.. class:: HasMeta

    Filters out instances that don't have a meta attribute with the given id.

    .. attribute:: id

    The id of the meta attribute to look for.

This is filter is especially useful with instances from basket format and
their optional meta attributes. If they come, for example,
from a text mining domain, we can use it to get the documents that contain a
specific word:

.. literalinclude:: code/filterm.py
    :lines: 3-5

In this example all instances that contain the word "surprise" are selected.
It does so by searching the :obj:`~Orange.data.Domain` for a meta attribute
named "suprise" present in the instance. This is an optional attribute that
does not necessarily appear in all instances. This filter can be used in
other situations involving meta values that appear only in some instances.
The corresponding attributes do not need to be registered in the domain.

Filtering by value
------------------

Single values
=============

.. class:: SameValue

    This is a fast filter for selecting instances with particular value of a
    feature.

    .. attribute:: position

    Index of feature in the :obj:`~Orange.data.Domain`. Method `index`
    provided by :obj:`~Orange.data.Domain` can be used to retrieve the
    position of a feature.

    .. attribute:: value

    Features's value.

Continuous features
===================

:obj:`ValueFilter` provides different methods for filtering values of
countinuous features: :obj:`ValueFilter.Equal`,
:obj:`ValueFilter.Less`, :obj:`ValueFilter.LessEqual`,
:obj:`ValueFilter.Greater`, :obj:`ValueFilter.GreaterEqual`,
:obj:`ValueFilter.Between`, :obj:`ValueFilter.Outside`.

In the following example two different filters are used:
:obj:`ValueFilter.GreaterEqual`, which needs only one parameter and
:obj:`ValueFilter.Between`, which needs to be defined by two parameters.

.. literalinclude:: code/filterv.py
    :lines: 52, 75-83

Multiple values and features
============================

:obj:`~Orange.data.filter.Values` performs a similar function as
:obj:`~Orange.data.filter.SameValue`, but can handle conjunctions and
disjunctions of more complex conditions.

.. class:: Values

    .. attribute:: conditions

    A list of :obj:`~Orange.data.filter.ValueFilterList` that contains
    conditions. Elements must be objects of type
    :obj:`~Orange.data.filter.ValueFilterDiscrete` for discrete and
    :obj:`~Orange.data.filter.ValueFilterContinuous` for continuous
    attributes; both are derived from
    :obj:`Orange.data.filter.ValueFilter`.

    .. attribute:: conjunction

    Decides whether the filter will compute conjunction or disjunction of
    conditions. If :obj:`True`, instance is accepted if no values are
    rejected. If :obj:`False`, instance is accepted if at least one value is
    accepted.

.. class:: ValueFilter

    .. attribute:: position

    Indicates the posiiton of the checked feature (similar to
    :obj:`Orange.data.filter.SameValue`).

    .. attribute:: accept_special

    Determines whether undefined values are accepted (1),
    rejected (0) or simply ignored (-1, default).

.. class:: ValueFilterDiscrete

    .. attribute:: values

    An immutable :obj:`list` that contains objects of type
    :obj:`~Orange.data.Value`, with values to accept.

.. class:: ValueFilterContinous

    .. attribute:: min

    Lower bound of values to consider.

    .. attribute:: max

    Upper bound of values to consider.

    .. attribute:: outside

    Indicates whether instances outside the interval should be accepted.
    Defaults to :obj:`False`.

.. literalinclude:: code/filter.py
    :lines: 68-82

This script selects instances whose age is "young" or "presbyopic" and
which are "astigmatic". Unknown values are ignored. If value for one of the
two features is missing, only the other is checked. If both are missing,
instance is accepted.

The filter is first constructed and assigned a domain. Then both
conditions are appended to the filter's
:obj:`~Orange.data.filter.Values.conditions` field. Both are of type
:obj:`~Orange.data.filter.ValueFilterDiscrete`, since the two attributes are
discrete. Position of the attribute is obtained the same way as for
:obj:`~Orange.data.filter.SameValue` described above.

The list of conditions can also be given to a filter constructor. The
following filter will accept examples whose age is "young" or "presbyopic"
or who are astigmatic (`conjunction = 0`). For contrast from
above filter, unknown age is not acceptable (but examples with unknown age
can still be accepted if they are astigmatic). Meanwhile,
examples with unknown astigmatism are always accepted.

.. literalinclude:: code/filter.py
    :lines: 129-141
