
class Tune1Parameter:
    returnNone=0
    returnParameters=1
    returnLearner=2
    returnClassifier=3
    
    def __init__(self, **keyw):
        self.__dict__=keyw

    def findobj(self, name):
        import string
        names=string.split(name, ".")
        lastobj=self.object
        for i in names[:-1]:
            lastobj=getattr(lastobj, i)
        return lastobj, names[-1]
        
    def __call__(self, table, weight, verbose=0):
        import types, whrandom
        import orange, orngTest, orngStat
        
        if (type(self.parameter)==types.ListType) or (type(self.parameter)==types.TupleType):
            to_set=[self.findobj(ld) for ld in self.parameter]
        else:
            to_set=[self.findobj(self.parameter)]

        cvind = orange.MakeRandomIndicesCV(table, 5)
        bestres = None

        evaluate = getattr(self, "evaluate", orngStat.CA)
        compare = getattr(self, "compare", cmp)

        # now, we need to set <lastobj>.<fieldname> to values in self.parameter[1]
        for par in self.values:
            for i in to_set:
                setattr(i[0], i[1], par)
            # self.evaluate could be, for example, orngEval.CA
            res=evaluate(orngTest.testWithIndices([self.object], (table, weight), cvind))
            if verbose:
                print par, res
            if not bestres:
                bestres, bestpar, wins = res, par, 1
            else:
                r=compare(res, bestres)
                if (r>0):
                    bestres, bestpar, wins = res, par, 1
                elif (r==0):
                    if not whrandom.randint(0, wins):
                        bestres, bestpar = res, par
                    wins=wins+1

        returnWhat = getattr(self, "returnWhat", Tune1Parameter.returnNone)

        for i in to_set:
            setattr(i[0], i[1], bestpar)

        if returnWhat==Tune1Parameter.returnNone:
            return None
        elif returnWhat==Tune1Parameter.returnParameters:
            return bestpar
        elif returnWhat==Tune1Parameter.returnLearner:
            return self.object
        else:
            return self.object(table)
