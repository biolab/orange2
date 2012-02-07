"""
=======================
Counters (``counters``)
=======================

.. index:: misc
.. index::
     single: misc; counters

:class:`Orange.misc.counters` contains a bunch of classes that generate sequences of various kinds.

"""

class BooleanCounter:
    """
    A class which represents a boolean counter. The constructor is given the number of bits and during the
    iteration the counter returns a list of that length with 0 and 1's in it.

    One way to use the counter is within a for-loop:

    >>> for r in BooleanCounter(3):
    ...    print r
    [0, 0, 0]
    [0, 0, 1]
    [0, 1, 0]
    [0, 1, 1]
    [1, 0, 0]
    [1, 0, 1]
    [1, 1, 0]
    [1, 1, 1]

    You can also call it manually.

    >>> r = BooleanCounter(3)
    >>> r.next()
    [0, 0, 0]
    >>> r.next()
    [0, 0, 1]
    >>> r.next()
    [0, 1, 0]

    .. attribute:: state
    
        The current counter state (the last result of a call to next) is also stored as attribute attribute.

    """
    
    def __init__(self, bits):
        """
            :param bits: Number of bits.
            :type bits: int
        """
        self.bits = bits
        self.state = None

    def __iter__(self):
        if self.state:
            return self
        else:
            return BooleanCounter(self.bits)
        
    def next(self):
        """Return the next state of the counter."""
        if self.state:
            for bit in range(self.bits-1, -1, -1):
                self.state[bit] = (self.state[bit]+1) % 2
                if self.state[bit]:
                    break
            else:
                self.state = None
        else:
            self.state = [0]*self.bits
        if not self.state:
            raise StopIteration, "BooleanCounter: counting finished"
        return self.state

class LimitedCounter:
    """
    This class is similar to :class:`~Orange.misc.counters.BooleanCounter` except that the digits do not count
    from 0 to 1, but to the limits that are specified individually for each digit.

    >>> for t in LimitedCounter([3, 5]):
    ...     print t
    [0, 0]
    [0, 1]
    [0, 2]
    [0, 3]
    [0, 4]
    [1, 0]
    [1, 1]
    [1, 2]
    [1, 3]
    [1, 4]
    [2, 0]
    [2, 1]
    [2, 2]
    [2, 3]
    [2, 4]

    .. attribute:: state

        The current counter state (the last result of a call to next) is also stored as attribute attribute.
    """
    
    def __init__(self, limits):
        """
            :param limits: Domain size per bit position.
            :type limits: list
        """
        self.limits = limits
        self.state = None
        
    def __iter__(self):
        if self.state:
            return self
        else:
            return LimitedCounter(self.limits)

    def next(self):
        """Return the next state of the counter."""
        if self.state:
            i = len(self.limits)-1
            while (i>=0) and (self.state[i]==self.limits[i]-1):
                self.state[i] = 0
                i -= 1
            if i==-1:
                self.state = None
            else:
                self.state[i] += 1
        else:
            self.state = [0]*len(self.limits)
    
        if not self.state:
            raise StopIteration, "LimitedCounter: counting finished"

        return self.state

class MofNCounter:
    """
    Counter returns all consecutive subset lists of length ``m`` out of ``n`` where ``m`` <= ``n``.

    >>> for t in MofNCounter(3,7):
    ...     print t
    ...
    [0, 1, 2]
    [1, 2, 3]
    [2, 3, 4]
    [3, 4, 5]
    [4, 5, 6]

    .. attribute:: state

        The current counter state (the last result of a call to next) is also stored as attribute attribute.
    """
    
    def __init__(self, m, n):
        """
        :param m: Length of subset list.
        :type m: int

        :param n: Total length.
        :type n: int
        """
        if m > n:
            raise TypeError, "Number of selected items exceeds the number of items"
                
        self.state = None
        self.m = m
        self.n = n
                
    def __iter__(self):
        if self.state:
            return self
        else:
            return MofNCounter(self.m, self.n)
                
    def next(self):
        """Return the next state of the counter."""
        if self.state:
            m, n, state = self.m, self.n, self.state
            for place in range(m-1, -1, -1):
                if state[place] + m-1-place < n-1:
                    state[place] += 1
                    for place in range(place+1, m):
                        state[place] = state[place-1] + 1
                        break
                else:
                    self.state = None
                    raise StopIteration, "MofNCounter: counting finished"
        else:
            self.state = range(self.m)
        return self.state[:]
                         
class NondecreasingCounter:
    """
    Nondecreasing counter generates all non-decreasing integer sequences in which no numbers are skipped,
    that is, if n is in the sequence, the sequence also includes all numbers between 0 and n. For instance,
    [0, 0, 1, 0] is illegal since it decreases, and [0, 0, 2, 2] is illegal since it has 2 without having 1
    first. Or, with an example

    Nondecreasing counter generates all non-decreasing integer sequences in which no numbers are skipped,
    that is, if ``n`` is in the sequence, the sequence also includes all numbers between 0 and ``n``. For instance,
    [0, 0, 1, 0] is illegal since it decreases, and [0, 0, 2, 2] is illegal since it has 2 without having 1 first.
    Or, with an example

    >>> for t in NondecreasingCounter(4):
    ...     print t
    ...
    [0, 0, 0, 0]
    [0, 0, 0, 1]
    [0, 0, 1, 1]
    [0, 0, 1, 2]
    [0, 1, 1, 1]
    [0, 1, 1, 2]
    [0, 1, 2, 2]
    [0, 1, 2, 3]

    .. attribute:: state

        The current counter state (the last result of a call to next) is also stored as attribute attribute.
    """
    def __init__(self, places):
        """
            :param places: Number of places.
            :type places: int
        """
        self.state=None
        self.subcounter=None
        self.places=places

    def __iter__(self):
        if self.state:
            return self
        else:
            return NondecreasingCounter(self.places)

    def next(self):
        """Return the next state of the counter."""
        if not self.subcounter:
            self.subcounter=BooleanCounter(self.places-1)
        if self.subcounter.next():
            self.state=[0]
            for add_one in self.subcounter.state:
                self.state.append(self.state[-1]+add_one)
        else:
            self.state=None
        if not self.state:
            raise StopIteration, "NondecreasingCounter: counting finished"
        return self.state


class CanonicFuncCounter:
    """
    Returns all sequences of a given length where no numbers are skipped (see below) and none of
    the generated sequence is equal to another if only the labels are changed. For instance, [0, 2, 2, 1]
    and [1, 0, 0, 2] are considered equivalent: if we take the former and replace 0 by 1, 2
    by 0 and 1 by 2 we get the second list.

    The sequences generated are equivalent to all possible functions from a set of cardinality of the sequences length.

    >>> for t in CanonicFuncCounter(4):
    ...     print t
    ...
    [0, 0, 0, 0]
    [0, 0, 0, 1]
    [0, 0, 1, 0]
    [0, 0, 1, 1]
    [0, 0, 1, 2]
    [0, 1, 0, 0]
    [0, 1, 0, 1]
    [0, 1, 0, 2]
    [0, 1, 1, 0]
    [0, 1, 1, 1]
    [0, 1, 1, 2]
    [0, 1, 2, 0]
    [0, 1, 2, 1]
    [0, 1, 2, 2]
    [0, 1, 2, 3]

    .. attribute:: state

        The current counter state (the last result of a call to next) is also stored as attribute attribute.
    """
    def __init__(self, places):
        """
            :param places: Number of places.
            :type places: int
        """
        self.places = places
        self.state = None

    def __iter__(self):
        if self.state:
            return self
        else:
            return CanonicFuncCounter(self.places)

    def next(self):
        """Return the next state of the counter."""
        if self.state:
            i = self.places-1
            while (i>0) and (self.state[i]==max(self.state[:i])+1):
                self.state[i] = 0
                i -= 1
            if i:
                self.state[i] += 1
            else:
                self.state=None
        else:
            self.state = [0]*self.places
        if not self.state:
            raise StopIteration, "CanonicFuncCounter: counting finished"
        return self.state
