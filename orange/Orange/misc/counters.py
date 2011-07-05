"""
=======================
Counters (``counters``)
=======================

.. index:: misc
.. index::
   single: misc; counters
"""

class BooleanCounter:
  def __init__(self, bits):
    self.bits = bits
    self.state = None

  def __iter__(self):
    if self.state:
        return self
    else:
        return BooleanCounter(self.bits)
    
  def next(self):
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
  def __init__(self, limits):
    self.limits = limits
    self.state = None
    
  def __iter__(self):
    if self.state:
        return self
    else:
        return LimitedCounter(self.limits)

  def next(self):
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
    def __init__(self, m, n):
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
  def __init__(self, places):
    self.state=None
    self.subcounter=None
    self.places=places

  def __iter__(self):
    if self.state:
        return self
    else:
        return NondecreasingCounter(self.places)

  def next(self):
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
  def __init__(self, places):
    self.places = places
    self.state = None

  def __iter__(self):
    if self.state:
        return self
    else:
        return CanonicFuncCounter(self.places)

  def next(self):
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
