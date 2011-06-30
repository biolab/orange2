"""
=============
Distributions
=============

Class :obj:`Distribution` and derived classes are used for storing empirical
distributions of discrete and continuous variables.

.. class:: Distribution

    A base class for storing distributions of variable values. The class can
    store absolute or relative frequencies. Provides a convenience constructor
    which constructs instances of derived classes. ::

        >>> import Orange
        >>> data = Orange.data.Table("adult_sample")
        >>> disc = orange.statistics.distribution.Distribution("workclass", data)
        >>> print disc
        <685.000, 72.000, 28.000, 29.000, 59.000, 43.000, 2.000>
        >>> print type(disc)
        <type 'DiscDistribution'>

    The resulting distribution is of type :obj:`DiscDistribution` since variable
    `workclass` is discrete. The printed numbers are counts of examples that have particular
    attribute value. ::

        >>> workclass = data.domain["workclass"]
        >>> for i in range(len(workclass.values)):
        ...     print "%20s: %5.3f" % (workclass.values[i], disc[i])
                 Private: 685.000
        Self-emp-not-inc: 72.000
            Self-emp-inc: 28.000
             Federal-gov: 29.000
               Local-gov: 59.000
               State-gov: 43.000
             Without-pay: 2.000
            Never-worked: 0.000

    Distributions resembles dictionaries, supporting indexing by instances of
    :obj:`Orange.data.Value`, integers or floats (depending on the distribution
    type), and symbolic names (if :obj:`variable` is defined).

    For instance, the number of examples with `workclass="private"`, can be
    obtained in three ways::
    
        print "Private: ", disc["Private"]
        print "Private: ", disc[0]
        print "Private: ", disc[orange.Value(workclass, "Private")]

    Elements cannot be removed from distributions.

    Length of distribution equals the number of possible values for discrete
    distributions (if :obj:`variable` is set), the value with the highest index
    encountered (if distribution is discrete and :obj: `variable` is
    :obj:`None`) or the number of different values encountered (for continuous
    distributions).

    .. attribute:: variable

        Variable to which the distribution applies; may be :obj:`None` if not
        applicable.

    .. attribute:: unknowns

        The number of instances for which the value of the variable was
        undefined.

    .. attribute:: abs

        Sum of all elements in the distribution. Usually it equals either
        :obj:`cases` if the instance stores absolute frequencies or 1 if the
        stored frequencies are relative, e.g. after calling :obj:`normalize`.

    .. attribute:: cases

        The number of instances from which the distribution is computed,
        excluding those on which the value was undefined. If instances were
        weighted, this is the sum of weights.

    .. attribute:: normalized

        :obj:`True` if distribution is normalized.

    .. attribute:: randomGenerator

        A pseudo-random number generator used for method :obj:`random`.

    .. method:: __init__(variable[, data[, weightId=0]])

        Construct either :obj:`DiscDistribution` or :obj:`ContDistribution`,
        depending on the variable type. If the variable is the only argument, it
        must be an instance of :obj:`Orange.data.variable.Variable`. In that case,
        an empty distribution is constructed. If data is given as well, the
        variable can also be specified by name or index in the
        domain. Constructor then computes the distribution of the specified
        variable on the given data. If instances are weighted, the id of
        meta-attribute with weights can be passed as the third argument.

        If variable is given by descriptor, it doesn't need to exist in the
        domain, but it must be computable from given instances. For example, the
        variable can be a discretized version of a variable from data.

    .. method:: keys()

        Return a list of possible values (if distribution is discrete and
        :obj:`variable` is set) or a list encountered values otherwise.

    .. method:: values()

        Return a list of frequencies of values such as described above.

    .. method:: items()

        Return a list of pairs of elements of the above lists.

    .. method:: native()

        Return the distribution as a list (for discrete distributions) or as a
        dictionary (for continuous distributions)

    .. method:: add(value[, weight=1])

        Increase the count of the element corresponding to ``value`` by
        ``weight``.

        :param value: Value
        :type value: :obj:`Orange.data.Value`, string (if :obj:`variable` is set), :obj:`int` for discrete distributions or :obj:`float` for continuous distributions
        :param weight: Weight to be added to the count for ``value``
        :type weight: float

    .. method:: normalize()

        Divide the counts by their sum, set :obj:`normalized` to :obj:`True` and
        :obj:`abs` to 1. Attributes :obj:`cases` and :obj:`unknowns` are
        unchanged. This changes absoluted frequencies into relative.

    .. method:: modus()

        Return the most common value. If there are multiple such values, one is
        chosen at random, although the chosen value will always be the same for
        the same distribution.

    .. method:: random()

        Return a random value based on the stored empirical probability
        distribution. For continuous distributions, this will always be one of
        the values which actually appeared (e.g. one of the values from
        :obj:`keys`).

        The method uses :obj:`randomGenerator`. If none has been constructed or
        assigned yet, a new one is constructed and stored for further use.


.. class:: Discrete

    Stores a discrete distribution of values. The class differs from its parent
    class in having a few additional constructors.

    .. method:: __init__(variable)

        Construct an instance of :obj:`Discrete` and set the variable
        attribute.

        :param variable: A discrete variable
        :type variable: Orange.data.variable.Discrete

    .. method:: __init__(frequencies)

        Construct an instance and initialize the frequencies from the list, but
        leave `Distribution.variable` empty.

        :param frequencies: A list of frequencies
        :type frequencies: list

        Distribution constructed in this way can be used, for instance, to
        generate random numbers from a given discrete distribution::

            disc = Orange.statistics.distribution.Discrete([0.5, 0.3, 0.2])
            for i in range(20):
                print disc.random(),

        This prints out approximatelly ten 0's, six 1's and four 2's. The values
        can be named by assigning a variable::

            v = orange.EnumVariable(values = ["red", "green", "blue"])
            disc.variable = v

    .. method:: __init__(distribution)

        Copy constructor; makes a shallow copy of the given distribution

        :param distribution: An existing discrete distribution
        :type distribution: Discrete


.. class:: Continuous

    Stores a continuous distribution, that is, a dictionary-like structure with
    values and their frequencies.

    .. method:: __init__(variable)

        Construct an instance of :obj:`ContDistribution` and set the variable
        attribute.

        :param variable: A continuous variable
        :type variable: Orange.data.variable.Continuous

    .. method:: __init__(frequencies)

        Construct an instance of :obj:`Continuous` and initialize it from
        the given dictionary with frequencies, whose keys and values must be integers.

        :param frequencies: Values and their corresponding frequencies
        :type frequencies: dict

    .. method:: __init__(distribution)

        Copy constructor; makes a shallow copy of the given distribution

        :param distribution: An existing continuous distribution
        :type distribution: Continuous

    .. method:: average()

        Return the average value. Note that the average can also be
        computed using a simpler and faster classes from module
        :obj:`Orange.statistics.basic`.

    .. method:: var()

        Return the variance of distribution.

    .. method:: dev()

        Return the standard deviation.

    .. method:: error()

        Return the standard error.

    .. method:: percentile(p)

        Return the value at the `p`-th percentile.

        :param p: The percentile, must be between 0 and 100
        :type p: float
        :rtype: float

        For example, if `d_age` is a continuous distribution, the quartiles can
        be printed by ::

            print "Quartiles: %5.3f - %5.3f - %5.3f" % ( 
                 dage.percentile(25), dage.percentile(50), dage.percentile(75))

   .. method:: density(x)

        Return the probability density at `x`. If the value is not in
        :obj:`Distribution.keys`, it is interpolated.


.. class:: Gaussian

    A class imitating :obj:`Continuous` by returning the statistics and
    densities for Gaussian distribution. The class is not meant only for a
    convenient substitution for code which expects an instance of
    :obj:`Distribution`. For general use, Python module :obj:`random`
    provides a comprehensive set of functions for various random distributions.

    .. attribute:: mean

        The mean value parameter of the Gauss distribution.

    .. attribute:: sigma

        The standard deviation of the distribution

    .. attribute:: abs

        The simulated number of instances; in effect, the Gaussian distribution
        density, as returned by method :obj:`density` is multiplied by
        :obj:`abs`.

    .. method:: __init__([mean=0, sigma=1])

        Construct an instance, set :obj:`mean` and :obj:`sigma` to the given
        values and :obj:`abs` to 1.

    .. method:: __init__(distribution)

        Construct a distribution which approximates the given distribution,
        which must be either :obj:`Continuous`, in which case its
        average and deviation will be used for mean and sigma, or and existing
        :obj:`GaussianDistribution`, which will be copied. Attribute :obj:`abs`
        is set to the given distribution's ``abs``.

    .. method:: average()

        Return :obj:`mean`.

    .. method:: dev()

        Return :obj:`sigma`.

    .. method:: var()

        Return square of :obj:`sigma`.

    .. method:: density(x)

        Return the density at point ``x``, that is, the Gaussian distribution
        density multiplied by :obj:`abs`.


Class distributions
===================

There is a convenience function for computing empirical class distributions from
data.

.. function:: getClassDistribution(data[, weightID=0])

    Return a class distribution for the given data.

    :param data: A set of instances.
    :type data: Orange.data.Table
    :param weightID: An id for meta attribute with weights of instances
    :type weightID: int
    :rtype: :obj:`Discrete` or :obj:`Continuous`, depending on the class type

Distributions of all variables
==============================

Distributions of all variables can be computed and stored in
:obj:`Domain`. The list-like object can be indexed by variable
indices in the domain, as well as by variables and their names.

.. class:: Domain

    .. method:: __init__(data[, weightID=0])

        Construct an instance with distributions of all discrete and continuous
        variables from the given data.

    :param data: A set of instances.
    :type data: Orange.data.Table
    :param weightID: An id for meta attribute with weights of instances
    :type weightID: int

The script below computes distributions for all attributes in the data and
prints out distributions for discrete and averages for continuous attributes. ::

    dist = Orange.statistics.distribution.Domain(data)

        for d in dist:
            if d.variable.varType == orange.VarTypes.Discrete:
                 print "%30s: %s" % (d.variable.name, d)
            else:
                 print "%30s: avg. %5.3f" % (d.variable.name, d.average())

The distribution for, say, attribute `age` can be obtained by its index and also
by its name::

    dist_age = dist["age"]

"""


from Orange.core import Distribution
from Orange.core import DiscDistribution as Discrete
from Orange.core import ContDistribution as Continuous
from Orange.core import GaussianDistribution as Gaussian

from Orange.core import DomainDistributions as Domain
