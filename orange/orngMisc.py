import whrandom, types

def getobjectname(x, default=""):
    if type(x)==types.StringType:
        return x
      
    for i in ["name", "shortDescription", "description", "func_doc", "func_name"]:
        try:
            if len(getattr(x, i, "")):
                return getattr(x, i)
        except Exception:
            pass

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
    
def classChecksum(dta):
    rgen=whrandom.whrandom()
    rgen.seed(1, 2, 3)
    sum=0
    for i in dta:
        sum+=int(i.getclass()) ^ rgen.randint(0, 255)
    return sum


def verbose_print(verb, s):
    if verb:
        print s


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
    if self.subcounter():
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
      raise IndexError, "CanonicFuncCounter: counting finished"
    
    return self.state



from whrandom import randint

class BestOnTheFly:
    def __init__(self, compare=cmp):
        self.compare=compare
        self.wins=0
        self.bestIndex, self.index = -1, -1
        self.best = None

    def candidate(self, x):
        self.index += 1
        if not self.wins:
            self.best=x
            self.wins=1
            self.bestIndex=self.index
            return 1
        else:
            cmpr=self.compare(x, self.best)
            if cmpr>0:
                self.best=x
                self.wins=1
                self.bestIndex=self.index
                return 1
            elif cmpr==0:
                self.wins=self.wins+1
                if not randint(0, self.wins-1):
                    self.best=x
                    self.bestIndex=self.index
                    return 1
        return 0

    def winner(self):
        return self.best

    def winnerIndex(self):
        if self.best:
            return self.bestIndex
        else:
            return None

def selectBest(x, compare=cmp):
    bs=BestOnTheFly(compare)
    for i in x:
        bs.candidate(i)
    return bs.winner()

def selectBestIndex(x, compare=cmp):
    bs=BestOnTheFly(compare)
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
    start, stop = 0.0, 1.0
    if len(argw)==1:
        start=step=argw[0]
    elif len(argw)==2:
        stop, step = argw
    elif len(argw)==3:
        start, stop, step = argw
    else:
        raise AttributeError, "1-3 arguments expected"

    stop+=1e-10
    i=0
    res=[]
    print start, stop, step
    while 1:
        f=start+i*step
        print f
        if f>stop:
            break
        res.append(f)
        print res
        i+=1
    return res

def delimitedList(aList, aDelimiter):
    return reduce(lambda x,y, delim=aDelimiter: x+delim+y, a_list)

import re
cnre=re.compile(".*\.(?P<classname>\w*) instance at")
def getclassname(object):
    import exceptions
    on=cnre.search(`object`)
    try:
        return on.group("classname")
    except Exception:
        pass
    return None
