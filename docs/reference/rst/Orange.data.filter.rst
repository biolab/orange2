.. py:currentmodule:: Orange.data.filter

.. index:: filter

.. index::
   single: filtering; instance filtering

**********************
Filtering (``filter``)
**********************

Filters select subsets of instances. They are most typically used to
select data instances from a table, for example to drop all
instances that have no class value::

    filtered = Orange.data.filter.HasClassValue(data)

Despite this typical use, filters operate on individual instances, not
the entire data table: they can be called with an instance and return
``True`` are ``False`` to accept or reject the instances. Most
examples below use them like this for sake of demonstration.

An alternative way to apply a filter is to call
:obj:`Orange.data.Table.filter` on the data table.

All filters are derived from the base class :obj:`Filter`.

.. class:: Filter

    Abstract base class for filters.

    .. attribute:: negate

        Inverts the selection. Defaults to :obj:`False`.

    .. attribute:: domain

        Domain to which data instances are converted before checking.

    .. method:: __call__(instance)

        Check whether the instance matches the filter's criterion and
        return either ``True`` or ``False``.

    .. method:: __call__(data)

        Return a new data table containing the instances that match
        the criterion.



Filtering missing data
----------------------

.. class:: IsDefined

    Selects instances for which all feature values are defined.

    .. attribute:: check

	A list of ``bool``'s specifying which features to check. Each
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


.. class:: HasClassValue

    Selects instances with defined class value. Setting
    :obj:`~Orange.data.filter.Filter.negate` inverts the selection and
    chooses examples with unknown class.

.. literalinclude:: code/filter.py
    :lines: 9, 49-55


.. class:: HasMeta

    Filters out instances that do not have a meta attribute with the given id.

    .. attribute:: id

        The id of the meta attribute to look for.

This is filter is especially useful with instances from basket files,
which have optional meta attributes. If they come, for example, from a
text mining domain, we can use it to get the documents that contain a
specific word:

.. literalinclude:: code/filterm.py
    :lines: 3, 5

Random filter
-------------

.. class:: Random

    Accepts an instance with a given probability.

    .. attribute:: prob

        Probability for accepting an instance.

    .. attribute:: random_generator

        The random number generator used for making selections. If not
        set before filtering, a new generator is constructed and
        stored here for later use. If the attribute is set to an
        integer, Orange constructs a random generator and uses the
        integer as a seed.

.. literalinclude:: code/filter.py
    :lines: 12-14

The output is::

    1 0 0 0 1 1 0 1 0 1

Although the probability of selecting an instance is set to 0.7, the
filter accepted five out of ten instances since the decision is made for each instance separately. To select exactly 70 % of instance (except for a rounding error), use :obj:`~Orange.data.sample.SubsetIndices2`.

Setting the random generator ensures that the filter will always
select the same instances. Setting `random_generator=24` is a shortcut
for `random_generator = Orange.misc.Random(initseed=24)`.


Filtering by single features
----------------------------

.. class:: SameValue

    Fast filter for selecting instances with particular value of a
    feature.

    .. attribute:: position

        Index of feature in the :obj:`~Orange.data.Domain` as returned
        by :obj:`Orange.data.Domain.index`.

    .. attribute:: value

        Features's value.

The following example selects instances with age="young" from data set
lenses:

.. literalinclude:: code/filter.py
    :lines: 58-64


``data.domain.``:obj:`~Orange.data.Domain.features` behaves as a list and provides method
`index`, which is used to retrieve the position of feature `age`. Feature
`age` is also used to construct a :obj:`~Orange.data.Value`.


Filtering by multiple features
------------------------------

:obj:`~Orange.data.filter.Values` filters by values of multiple
features presented as subfilters derived from
:obj:`Orange.data.filter.ValueFilter`.

.. class:: Values

    .. attribute:: conditions

        A list of conditions described by instances of classes derived from 
        :obj:`Orange.data.filter.ValueFilter`.

    .. attribute:: conjunction

        Indicates whether the filter computes conjunction or disjunction
        of conditions. If ``True``, instance is accepted if no
        values are rejected. If ``False``, instance is accepted if
        at least one value is accepted.

The attribute :obj:`conditions` contains subfilter instances of the following classes.

.. class:: ValueFilter

    The abstract base class for subfilters.

    .. attribute:: position

        The position of the feature in the domain (as returned by, for
        instance, :obj:`Orange.data.Domain.index`).

    .. attribute:: accept_special

        Determines whether undefined values are accepted (``1``),
        rejected (``0``) or ignored (``-1``, default).

.. class:: ValueFilterDiscrete

    Subfilter for values of discrete features.

    .. attribute:: values

        An list of accepted values with elements of type
        :obj:`~Orange.data.Value`.

.. class:: ValueFilterContinous

    Subfilter for values of continuous features.

    .. attribute:: min / ref

        Lower bound of the interval (``min`` and ``ref`` are aliases
        for the same attribute).

    .. attribute:: max

        Upper bound of the interval.

    .. attribute:: oper

        Comparison operator; should be one of the following:
        :obj:`ValueFilter.Equal`, :obj:`ValueFilter.Less`,
        :obj:`ValueFilter.LessEqual`, :obj:`ValueFilter.Greater`,
        :obj:`ValueFilter.GreaterEqual`, :obj:`ValueFilter.Between`,
        :obj:`ValueFilter.Outside`.

    Attributes ``min`` and ``max`` define the interval for
    operators :obj:`ValueFilter.Between` and :obj:`ValueFilter.Outside`
    and ``ref`` (which is the same as ``min``) for the others.


.. class:: ValueFilterString

    Subfilter for values of discrete features.

    .. attribute:: min / ref

        Lower bound of the interval (``min`` and ``ref`` are aliases
        for the same attribute.

    .. attribute:: max

        Upper bound of the interval.

    .. attribute:: oper

        Comparison operator; should be one of the following:
        :obj:`ValueFilter.Equal`, :obj:`ValueFilter.Less`,
        :obj:`ValueFilter.LessEqual`, :obj:`ValueFilter.Greater`,
        :obj:`ValueFilter.GreaterEqual`, :obj:`ValueFilter.Between`,
        :obj:`ValueFilter.Outside`, :obj:`Contains`,
        :obj:`NotContains`, :obj:`BeginsWith`, :obj:`EndsWith`.
    
    .. attribute:: case_sensitive

        Tells whether the comparisons are case sensitive. Default is ``True``.

    Attributes ``min`` and ``max`` define the interval for
    operators :obj:`ValueFilter.Between` and :obj:`ValueFilter.Outside`
    and ``ref`` (which is the same as ``min``) for the others.

.. class:: ValueFilterStringList

    Accepts string values from the list.

    .. attribute:: values

        A list of accepted strings.

    .. attribute:: case_sensitive

        Tells whether the comparisons are case sensitive. Default is ``True``.


The following script selects instances whose age is "young" or "presbyopic" and
which are "astigmatic". Unknown values are ignored. If value for one of the
two features is missing, only the other is checked. If both are missing,
instance is accepted.

.. literalinclude:: code/filter.py
    :lines: 68-82

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

Composition of filters
----------------------

Filters can be combined into conjuctions or disjunctions using the following descendants of :obj:`Filter`. It is possible to build hierarchies of filters (e.g. disjunction of conjuctions).

.. class:: FilterConjunction

    Reject the instance if any of the combined filters rejects
    it. Conjunction can be negated using the inherited
    :obj:``~Filter.negate`` flag.

    .. attribute:: filters

        A list of filters (instances of :obj:`Filter`)

.. class:: FilterDisjunction

    Accept the instance if any of the combined filters accepts
    it. Disjunction can be negated using the inherited
    :obj:``~Filter.negate`` flag.
    
    .. attribute:: filters

        A list of filters (instances of :obj:`Filter`)
