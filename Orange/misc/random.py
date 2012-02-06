"""

.. index: Random number generator

***********************
Random number generator
***********************

:obj:`RandomGenerator` uses the 
`Mersenne twister <http://en.wikipedia.org/wiki/Mersenne_twister>`_ algorithm
to generate random numbers.

::

    >>> import Orange
    >>> rg = Orange.misc.random.RandomGenerator(42)
    >>> rg(10)
    4
    >>> rg(10)
    7
    >>> rg.uses  # We called rg two times.
    2
    >>> rg.reset()
    >>> rg(10)
    4
    >>> rg(10)
    7
    >>> rg.uses
    2


.. class:: RandomGenerator(initseed)

    :param initseed: Seed used for initializing the random generator.
    :type initseed: int

    .. method:: __call__(n)

        Return a random integer R such that 0 <= R < n.

        :type n: int

    .. method:: reset([initseed])

        Reinitialize the random generator with `initseed`. If `initseed`
        is not given use the existing value of attribute `initseed`.

    .. attribute:: uses
        
        The number of times the generator was called after
        initialization/reset.
    
    .. attribute:: initseed

        Random seed.

Two examples or random number generator uses found in the documentation
are :obj:`Orange.evaluation.testing` and :obj:`Orange.data.Table`.

"""

from Orange.core import RandomGenerator
