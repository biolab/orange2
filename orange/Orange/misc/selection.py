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
