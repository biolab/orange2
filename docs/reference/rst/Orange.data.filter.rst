.. py:currentmodule:: Orange.data.filter

.. index:: filter

.. index::
   single: filtering; instance filtering

**********************
Filtering (``filter``)
**********************

Filters select subsets of instances. Consider the following
example that selects instances with age="young" from data set lenses:

.. literalinclude:: code/filter.py
    :lines: 58-64

Output::

    Young instances
    ['young', 'myope', 'no', 'reduced', 'none']
    ['young', 'myope', 'no', 'normal', 'soft']
    ['young', 'myope', 'yes', 'reduced', 'none']
    ['young', 'myope', 'yes', 'normal', 'hard']
    ['young', 'hypermetrope', 'no', 'reduced', 'none']
    ['young', 'hypermetrope', 'no', 'normal', 'soft']
    ['young', 'hypermetrope', 'yes', 'reduced', 'none']
    ['young', 'hypermetrope', 'yes', 'normal', 'hard']

``data.domain.``:obj:`~Orange.data.Domain.features` behaves as a list and provides method
`index`, which is used to retrieve the position of feature `age`. Feature
`age` is also used to construct a :obj:`~Orange.data.Value`.


Filters operator on individual instances, not the entire data table,
and are limited to accepting or rejecting instances. All filters are derived from the base class :obj:`Filter`.

.. class:: Filter

    .. attribute:: negate

        Inverts the selection. Defaults to :obj:`False`.

    .. attribute:: domain

        Domain to which examples are converted prior to checking.

    .. method:: __call__(instance)

        Check whether the instance matches the filter's criterion and
        return either :obj:`True` or :obj:`False`.

    .. method:: __call__(table)

        Return a new data table containing the instances that match
        the criterion.

        An alternative way to apply a filter is to call
        :obj:`~Orange.data.Table.filter` on the data table.

Random filter
-------------

.. class:: Random

    Accepts an instance with a given probability.

    .. attribute:: prob

        Probability for accepting an instance.

    .. attribute:: random_generator

        The random number generator used for making selections. If not
        set before filtering, a new generator is constructed and
        stored here for later use.

.. literalinclude:: code/filter.py
    :lines: 12-14

The output is::

    1 0 0 0 1 1 0 1 0 1

Although the probability of selecting an instance is set to 0.7, the
filter accepted five out of ten instances since the decision is made for each instance separately. To select exactly 70 % of instance (except for a rounding error), use :obj:`~Orange.data.sample.SubsetIndices2`.

Setting the random generator ensures that the filter will always
select the same instances. Setting `randomGenerator=24` is a shortcut
for `randomGenerator = Orange.misc.Random(initseed=24)` or
`randomGenerator = Orange.misc.Random(initseed=24)`.

To select a subset of instances instead of calling the filter for each
individual example, call::

    data70 = randomfilter(data)


Filtering instances with missing data
-------------------------------------

.. class:: IsDefined

    Selects instances for which all feature values are defined.  By
    default, the filter checks all features; this can be changed by
    setting the attribute :obj:`check`. The filter does not check meta
    attributes.

    .. attribute:: check

	A list of ``bool``s specifying which features to check. Each
	element corresponds to a feature in the domain. By default,
	:obj:`check` is ``None``, meaning that all features are
	checked. The list is initialized to a list of ``True`` when
	the filter's :obj:`~Orange.data.filter.Filter.domain` is set,
	unless the list already exists. The list can be indexed by
	ordinary integers (for example, `check[0]`); if
	:obj:`~Orange.data.filter.Filter.domain` is set, feature names
	or descriptors can also be used as indices.

    .. literalinclude:: code/filter.py
        :lines: 9, 20-40


.. class:: HasClass

    Selects instances with defined class value. Setting
    :obj:`~Orange.data.filter.Filter.negate` to inverts the selection.


    .. literalinclude:: code/filter.py
        :lines: 9, 49-55


.. class:: HasMeta

    Filters out instances that do not have a meta attribute with the given id.

    .. attribute:: id

        The id of the meta attribute to look for.

    This is filter is especially useful with instances from basket
    files, which have optional meta attributes. If they come, for
    example, from a text mining domain, we can use it to get the
    documents that contain a specific word:

    .. literalinclude:: code/filterm.py
        :lines: 3, 5


Filtering by values
-------------------

Single values
.............

.. class:: SameValue

    Fast filter for selecting instances with particular value of a
    feature.

    .. attribute:: position

        Index of feature in the :obj:`~Orange.data.Domain`, as
        returned by :obj:`Orange.data.Domain.index`.

    .. attribute:: value

        Features's value.


Continuous features
...................

:obj:`Orange.data.filter.Values` provides different methods for
filtering values of countinuous features: :obj:`ValueFilter.Equal`,
:obj:`ValueFilter.Less`, :obj:`ValueFilter.LessEqual`,
:obj:`ValueFilter.Greater`, :obj:`ValueFilter.GreaterEqual`,
:obj:`ValueFilter.Between`, :obj:`ValueFilter.Outside`.

In the following example two different filters are used:
:obj:`ValueFilter.GreaterEqual`, which needs only one parameter and
:obj:`ValueFilter.Between`, which needs two.

.. literalinclude:: code/filterv.py
    :lines: 52, 75-83


Multiple values and features
............................

:obj:`~Orange.data.filter.Values` filters by values of multuple
features and can compute conjunctions and disjunctions of more complex
conditions.

.. class:: Values

    .. attribute:: conditions

        A list of conditions described by instances of
        :obj:`~Orange.data.filter.ValueFilterDiscrete` for discrete
        features and :obj:`~Orange.data.filter.ValueFilterContinuous`
        for continuous ones; both are derived from
        :obj:`Orange.data.filter.ValueFilter`.

    .. attribute:: conjunction

        Decides whether the filter computes conjunction or disjunction
        of conditions. If ``True``, instance is accepted if no
        values are rejected. If ``False``, instance is accepted if
        at least one value is accepted.

.. class:: ValueFilter

    The abstract base class for filters for discrete and continuous features.

    .. attribute:: position

        The position of the checked feature (as returned by, for
        instance, :obj:`Orange.data.Domain.index`).

    .. attribute:: accept_special

        Determines whether undefined values are accepted (``1``),
        rejected (``0``) or ignored (``-1``, default).

.. class:: ValueFilterDiscrete

    .. attribute:: values

        An immutable ``list`` that contains objects of type
        :obj:`~Orange.data.Value`, with values to accept.

.. class:: ValueFilterContinous

    .. attribute:: min

        Lower bound of values to consider.

    .. attribute:: max

        Upper bound of values to consider.

    .. attribute:: outside

        Indicates whether instances outside the interval should be
        accepted.  Defaults to :obj:`False`.

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

