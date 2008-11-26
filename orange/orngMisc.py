import random, types

def getobjectname(x, default=""):
    if type(x)==types.StringType:
        return x
      
    for i in ["name", "shortDescription", "description", "func_doc", "func_name"]:
        if getattr(x, i, ""):
            return getattr(x, i)

    if hasattr(x, "__class__"):
        r = repr(x.__class__)
        if r[1:5]=="type":
            return str(x.__class__)[7:-2]
        elif r[1:6]=="class":
            return str(x.__class__)[8:-2]

    return default


def demangleExamples(x):
    if type(x)==types.TupleType:
        return x
    else:
        return x, 0
    

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



import random

class BestOnTheFly:
    def __init__(self, compare=cmp, seed = 0, callCompareOn1st = False):
        self.randomGenerator = random.Random(seed)
        self.compare=compare
        self.wins=0
        self.bestIndex, self.index = -1, -1
        self.best = None
        self.callCompareOn1st = callCompareOn1st

    def candidate(self, x):
        self.index += 1
        if not self.wins:
            self.best=x
            self.wins=1
            self.bestIndex=self.index
            return 1
        else:
            if self.callCompareOn1st:
                cmpr=self.compare(x[0], self.best[0])
            else:
                cmpr=self.compare(x, self.best)
            if cmpr>0:
                self.best=x
                self.wins=1
                self.bestIndex=self.index
                return 1
            elif cmpr==0:
                self.wins=self.wins+1
                if not self.randomGenerator.randint(0, self.wins-1):
                    self.best=x
                    self.bestIndex=self.index
                    return 1
        return 0

    def winner(self):
        return self.best

    def winnerIndex(self):
        if self.best is not None:
            return self.bestIndex
        else:
            return None

def selectBest(x, compare=cmp, seed = 0, callCompareOn1st = False):
    bs=BestOnTheFly(compare, seed, callCompareOn1st)
    for i in x:
        bs.candidate(i)
    return bs.winner()

def selectBestIndex(x, compare=cmp, seed = 0, callCompareOn1st = False):
    bs=BestOnTheFly(compare, seed, callCompareOn1st)
    for i in x:
        bs.candidate(i)
    return bs.winnerIndex()

def compare2_firstBigger(x, y):
    return cmp(x[0], y[0])

def compare2_firstSmaller(x, y):
    return -cmp(x[0], y[0])

def compare2_lastBigger(x, y):
    return cmp(x[-1], y[-1])

def compare2_lastSmaller(x, y):
    return -cmp(x[-1], y[-1])

def compare2_bigger(x, y):
    return cmp(x, y)

def compare2_smaller(x, y):
    return -cmp(x, y)


def frange(*argw):
    start, stop, step = 0.0, 1.0, 0.1
    if len(argw)==1:
        start=step=argw[0]
    elif len(argw)==2:
        stop, step = argw
    elif len(argw)==3:
        start, stop, step = argw
    elif len(argw)>3:
        raise AttributeError, "1-3 arguments expected"

    stop+=1e-10
    i=0
    res=[]
    while 1:
        f=start+i*step
        if f>stop:
            break
        res.append(f)
        i+=1
    return res


verbose = 0

def printVerbose(text, *verb):
    if len(verb) and verb[0] or verbose:
        print text

import sys
class consoleProgressBar(object):
    def __init__(self, title="", charwidth=40, step=1):
        self.title = title
        self.charwidth = charwidth
        self.step = step
        self.currstring = ""
        self.state = 0

    def clear(self, i=-1):
        if sys.stdout.isatty():
            sys.stdout.write("\b" * (i if i != -1 else len(self.currstring)))
        else:
            sys.stdout.seek(-i if i != -1 else -len(self.currstring), 2)

    def getstring(self):
        progchar = int(round(float(self.state) * (self.charwidth - 5) / 100.0))
        return self.title + "=" * (progchar) + ">" + " " * (self.charwidth - 5 - progchar) + "%3i" % int(round(self.state)) + "%"

    def printline(self, string):
        self.clear()
        sys.stdout.write(string)
        sys.stdout.flush()
        self.currstring = string

    def __call__(self, newstate=None):
        if newstate == None:
            newstate = self.state + self.step
        if int(newstate) != int(self.state):
            self.state = newstate
            self.printline(self.getstring())
        else:
            self.state = newstate

    def finish(self):
        self.__call__(100)
        sys.stdout.write("\n")
        
