import random

class BestOnTheFly:
    """
    Finds the optimal object in a sequence of objects. The class is fed the
    candidates one by one, and remembers the winner. It can thus be used by
    methods that generate different solutions to a problem and need to
    select the optimal one, but do not want to store them all.
    
    :param compare: compare function.
    :param seed: If not given, a random seed of 0 is used to ensure that\
    the same experiment always gives the same results, despite\
    pseudo-randomness.random seed.
    :type seed: int
    :param callCompareOn1st: If set, :obj:`BestOnTheFly` will suppose\
    that the candidates are lists are tuples, and it will call compare\
    with the first element of the tuple.
    :type callCompareOn1st: bool
    """
    
    def __init__(self, compare=cmp, seed = 0, callCompareOn1st = False):
        self.randomGenerator = random.Random(seed)
        self.compare=compare
        self.wins=0
        self.bestIndex, self.index = -1, -1
        self.best = None
        self.callCompareOn1st = callCompareOn1st

    def candidate(self, x):
        """Add new candidate.
        
        :param x: new candidate.
        :type x: object"""
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
        """Return (currently) optimal object. This function can be called
        any number of times, even when the candidates are still coming.
        
        :rtype: object"""
        return self.best

    def winnerIndex(self):
        """Return the index of the optimal object within the sequence of
        the candidates.
        
        :rtype: int"""
        if self.best is not None:
            return self.bestIndex
        else:
            return None

def selectBest(x, compare=cmp, seed = 0, callCompareOn1st = False):
    """Return the optimal object from list x. The function is used if the candidates
    are already in the list, so using the more complicated :obj:`BestOnTheFly` directly is
    not needed.

    To demonstrate the use of :obj:`BestOnTheFly` see the implementation of
    :obj:`selectBest`::
    
      def selectBest(x, compare=cmp, seed = 0, callCompareOn1st = False):
          bs=BestOnTheFly(compare, seed, callCompareOn1st)
          for i in x:
              bs.candidate(i)
          return bs.winner()

    :param x: list of existing candidates.
    :type x: list
    :param compare: compare function.
    :param seed: If not given, a random seed of 0 is used to ensure that\
    the same experiment always gives the same results, despite\
    pseudo-randomness.random seed.
    :type seed: int
    :param callCompareOn1st: If set, :obj:`BestOnTheFly` will suppose\
    that the candidates are lists are tuples, and it will call compare\
    with the first element of the tuple.
    :type callCompareOn1st: bool
    :rtype: object"""
    bs=BestOnTheFly(compare, seed, callCompareOn1st)
    for i in x:
        bs.candidate(i)
    return bs.winner()

def selectBestIndex(x, compare=cmp, seed = 0, callCompareOn1st = False):
    """Similar to :obj:`selectBest` except that it doesn't return the best object
    but its index in the list x."""
    bs=BestOnTheFly(compare, seed, callCompareOn1st)
    for i in x:
        bs.candidate(i)
    return bs.winnerIndex()

# def compare2_firstBigger(x, y):
def compareFirstBigger(x, y):
    """Function takes two lists and compares first elements.
    
    :param x: list of values.
    :type x: list
    :param y: list of values.
    :type y: list
    :rtype:  cmp(x[0], y[0])"""
    return cmp(x[0], y[0])

#def compare2_firstSmaller(x, y):
def compareFirstSmaller(x, y):
    """Function takes two lists and compares first elements.
    
    :param x: list of values.
    :type x: list
    :param y: list of values.
    :type y: list
    :rtype:  -cmp(x[0], y[0])"""
    return -cmp(x[0], y[0])

#     def compare2_lastBigger(x, y):
def compareLastBigger(x, y):
    """Function takes two lists and compares last elements.
    
    :param x: list of values.
    :type x: list
    :param y: list of values.
    :type y: list
    :rtype:  cmp(x[0], y[0])"""
    return cmp(x[-1], y[-1])

#    def compare2_lastSmaller(x, y):
def compareLastSmaller(x, y):
    """Function takes two lists and compares last elements.
    
    :param x: list of values.
    :type x: list
    :param y: list of values.
    :type y: list
    :rtype:  -cmp(x[0], y[0])"""
    return -cmp(x[-1], y[-1])
    
#     def compare2_bigger(x, y):
def compareBigger(x, y):
    """Function takes and compares two numbers.
    
    :param x: value.
    :type x: int
    :param y: value.
    :type y: int
    :rtype:  cmp(x, y)"""
    return cmp(x, y)
#     def compare2_smaller(x, y):
def compareSmaller(x, y):
    """Function takes and compares two numbers.
    
    :param x: value.
    :type x: int
    :param y: value.
    :type y: int
    :rtype: cmp(x, y)"""
    return -cmp(x, y)