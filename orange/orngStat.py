import statc, operator, math
from operator import add
import orngMisc, orngTest


#### Private stuff

def log2(x):
    return math.log(x)/math.log(2)

def checkNonZero(x):
    if x==0.0:
        raise SystemError, "Cannot compute the score: no examples or sum of weights is 0.0."

def gettotweight(res):
    totweight = reduce(lambda x, y: x+y.weight, res.results, 0)
    if totweight==0.0:
        raise SystemError, "Cannot compute the score: sum of weights is 0.0."
    return totweight

def gettotsize(res):
    if len(res.results):
        return len(res.results)
    else:
        raise SystemError, "Cannot compute the score: no examples."


def splitByIterations(res):
    if res.numberOfIterations < 2:
        return [res]
        
    ress = [orngTest.ExperimentResults(1, res.classifierNames, res.classValues, res.weights, classifiers=res.classifiers, loaded=res.loaded)
            for i in range(res.numberOfIterations)]
    for te in res.results:
        ress[te.iterationNumber].results.append(te)
    return ress    


def classProbabilitiesFromRes(res, **argkw):
    probs = [0.0] * len(res.classValues)
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            probs[int(tex.actualClass)] += 1.0
        totweight = gettotsize(res)
    else:
        totweight = 0.0
        for tex in res.results:
            probs[tex.actualClass] += tex.weight
            totweight += tex.weight
        checkNonZero(totweight)
    return [prob/totweight for prob in probs]


def statisticsByFolds(stats, foldN, reportSE, iterationIsOuter):
    # remove empty folds, turn the matrix so that learner is outer
    if iterationIsOuter:
        if not stats:
            raise SystemError, "Cannot compute the score: no examples or sum of weights is 0.0."
        numberOfLearners = len(stats[0])
        stats = filter(lambda (x, fN): fN>0.0, zip(stats,foldN))
        stats = [ [x[lrn]/fN for x, fN in stats] for lrn in range(numberOfLearners)]
    else:
        stats = [ [x/Fn for x, Fn in filter(lambda (x, Fn): Fn > 0.0, zip(lrnD, foldN))] for lrnD in stats]

    if not stats:
        raise SystemError, "Cannot compute the score: no classifiers"
    if not stats[0]:
        raise SystemError, "Cannot compute the score: no examples or sum of weights is 0.0."
    
    if reportSE:
        return [(statc.mean(x), statc.sterr(x)) for x in stats]
    else:
        return [statc.mean(x) for x in stats]
    
def ME(res, **argkw):
    MEs = [0.0]*res.numberOfLearners

    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            MEs = map(lambda res, cls, ac = float(tex.actualClass):
                      res + abs(float(cls) - ac), MEs, tex.classes)
        totweight = gettotsize(res)
    else:
        for tex in res.results:
            MEs = map(lambda res, cls, ac = float(tex.actualClass), tw = tex.weight:
                       res + tw*abs(float(cls) - ac), MEs, tex.classes)
        totweight = gettotweight(res)

    return [x/totweight for x in MEs]

MAE = ME

#########################################################################
# PERFORMANCE MEASURES:
# Scores for evaluation of numeric predictions

def checkArgkw(dct, lst):
    """checkArgkw(dct, lst) -> returns true if any items have non-zero value in dct"""
    return reduce(lambda x,y: x or y, [dct.get(k, 0) for k in lst])

def regressionError(res, **argkw):
    """regressionError(res) -> regression error (default: MSE)"""
    if argkw.get("SE", 0) and res.numberOfIterations > 1:
        # computes the scores for each iteration, then averages
        scores = [[0.0] * res.numberOfIterations for i in range(res.numberOfLearners)]
        if argkw.get("norm-abs", 0) or argkw.get("norm-sqr", 0):
            norm = [0.0] * res.numberOfIterations

        nIter = [0]*res.numberOfIterations       # counts examples in each iteration
        a = [0]*res.numberOfIterations           # average class in each iteration
        for tex in res.results:
            nIter[tex.iterationNumber] += 1
            a[tex.iterationNumber] += float(tex.actualClass)
        a = [a[i]/nIter[i] for i in range(res.numberOfIterations)]

        if argkw.get("unweighted", 0) or not res.weights:
            # iterate accross test cases
            for tex in res.results:
                ai = float(tex.actualClass)
                nIter[tex.iterationNumber] += 1

                # compute normalization, if required
                if argkw.get("norm-abs", 0):
                    norm[tex.iterationNumber] += abs(ai - a[tex.iterationNumber])
                elif argkw.get("norm-sqr", 0):
                    norm[tex.iterationNumber] += (ai - a[tex.iterationNumber])**2

                # iterate accross results of different regressors
                for i, cls in enumerate(tex.classes):
                    if argkw.get("abs", 0):
                        scores[i][tex.iterationNumber] += abs(float(cls) - ai)
                    else:
                        scores[i][tex.iterationNumber] += (float(cls) - ai)**2
        else: # unweighted<>0
            raise SystemError, "weighted error scores with SE not implemented yet"

        if argkw.get("norm-abs") or argkw.get("norm-sqr"):
            scores = [[x/n for x, n in zip(y, norm)] for y in scores]
        else:
            scores = [[x/ni for x, ni in zip(y, nIter)] for y in scores]

        if argkw.get("R2"):
            scores = [[1.0 - x for x in y] for y in scores]

        if argkw.get("sqrt", 0):
            scores = [[math.sqrt(x) for x in y] for y in scores]

        return [(statc.mean(x), statc.std(x)) for x in scores]
        
    else: # single iteration (testing on a single test set)
        scores = [0.0] * res.numberOfLearners
        norm = 0.0

        if argkw.get("unweighted", 0) or not res.weights:
            a = sum([tex.actualClass for tex in res.results]) \
                / len(res.results)
            for tex in res.results:
                if argkw.get("abs", 0):
                    scores = map(lambda res, cls, ac = float(tex.actualClass):
                                 res + abs(float(cls) - ac), scores, tex.classes)
                else:
                    scores = map(lambda res, cls, ac = float(tex.actualClass):
                                 res + (float(cls) - ac)**2, scores, tex.classes)

                if argkw.get("norm-abs", 0):
                    norm += abs(tex.actualClass - a)
                elif argkw.get("norm-sqr", 0):
                    norm += (tex.actualClass - a)**2
            totweight = gettotsize(res)
        else:
            # UNFINISHED
            for tex in res.results:
                MSEs = map(lambda res, cls, ac = float(tex.actualClass),
                           tw = tex.weight:
                           res + tw * (float(cls) - ac)**2, MSEs, tex.classes)
            totweight = gettotweight(res)

        if argkw.get("norm-abs", 0) or argkw.get("norm-sqr", 0):
            scores = [s/norm for s in scores]
        else: # normalize by number of instances (or sum of weights)
            scores = [s/totweight for s in scores]

        if argkw.get("R2"):
            scores = [1.0 - s for s in scores]

        if argkw.get("sqrt", 0):
            scores = [math.sqrt(x) for x in scores]

        return scores

def MSE(res, **argkw):
    """MSE(res) -> mean-squared error"""
    return regressionError(res, **argkw)
    
def RMSE(res, **argkw):
    """RMSE(res) -> root mean-squared error"""
    argkw.setdefault("sqrt", True)
    return regressionError(res, **argkw)

def MAE(res, **argkw):
    """MAE(res) -> mean absolute error"""
    argkw.setdefault("abs", True)
    return regressionError(res, **argkw)

def RSE(res, **argkw):
    """RSE(res) -> relative squared error"""
    argkw.setdefault("norm-sqr", True)
    return regressionError(res, **argkw)

def RRSE(res, **argkw):
    """RRSE(res) -> root relative squared error"""
    argkw.setdefault("norm-sqr", True)
    argkw.setdefault("sqrt", True)
    return regressionError(res, **argkw)

def RAE(res, **argkw):
    """RAE(res) -> relative absolute error"""
    argkw.setdefault("abs", True)
    argkw.setdefault("norm-abs", True)
    return regressionError(res, **argkw)

def R2(res, **argkw):
    """R2(res) -> R-squared"""
    argkw.setdefault("norm-sqr", True)
    argkw.setdefault("R2", True)
    return regressionError(res, **argkw)

def MSE_old(res, **argkw):
    """MSE(res) -> mean-squared error"""
    if argkw.get("SE", 0) and res.numberOfIterations > 1:
        MSEs = [[0.0] * res.numberOfIterations for i in range(res.numberOfLearners)]
        nIter = [0]*res.numberOfIterations
        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
                ac = float(tex.actualClass)
                nIter[tex.iterationNumber] += 1
                for i, cls in enumerate(tex.classes):
                    MSEs[i][tex.iterationNumber] += (float(cls) - ac)**2
        else:
            raise SystemError, "weighted RMSE with SE not implemented yet"
        MSEs = [[x/ni for x, ni in zip(y, nIter)] for y in MSEs]
        if argkw.get("sqrt", 0):
            MSEs = [[math.sqrt(x) for x in y] for y in MSEs]
        return [(statc.mean(x), statc.std(x)) for x in MSEs]
        
    else:
        MSEs = [0.0]*res.numberOfLearners
        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
                MSEs = map(lambda res, cls, ac = float(tex.actualClass):
                           res + (float(cls) - ac)**2, MSEs, tex.classes)
            totweight = gettotsize(res)
        else:
            for tex in res.results:
                MSEs = map(lambda res, cls, ac = float(tex.actualClass), tw = tex.weight:
                           res + tw * (float(cls) - ac)**2, MSEs, tex.classes)
            totweight = gettotweight(res)

        if argkw.get("sqrt", 0):
            MSEs = [math.sqrt(x) for x in MSEs]
        return [x/totweight for x in MSEs]

def RMSE_old(res, **argkw):
    """RMSE(res) -> root mean-squared error"""
    argkw.setdefault("sqrt", 1)
    return MSE_old(res, **argkw)


#########################################################################
# PERFORMANCE MEASURES:
# Scores for evaluation of classifiers

def CA(res, reportSE = False, **argkw):
    if res.numberOfIterations==1:
        if type(res)==ConfusionMatrix:
            div = nm.TP+nm.FN+nm.FP+nm.TN
            checkNonZero(div)
            ca = [(nm.TP+nm.TN)/div]
        else:
            CAs = [0.0]*res.numberOfLearners
            if argkw.get("unweighted", 0) or not res.weights:
                totweight = gettotsize(res)
                for tex in res.results:
                    CAs = map(lambda res, cls: res+(cls==tex.actualClass), CAs, tex.classes)
            else:
                totweight = 0.
                for tex in res.results:
                    CAs = map(lambda res, cls: res+(cls==tex.actualClass and tex.weight), CAs, tex.classes)
                    totweight += tex.weight
            checkNonZero(totweight)
            ca = [x/totweight for x in CAs]
            
        if reportSE:
            return [(x, x*(1-x)/math.sqrt(totweight)) for x in ca]
        else:
            return ca
        
    else:
        CAsByFold = [[0.0]*res.numberOfIterations for i in range(res.numberOfLearners)]
        foldN = [0.0]*res.numberOfIterations

        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
                for lrn in range(res.numberOfLearners):
                    CAsByFold[lrn][tex.iterationNumber] += (tex.classes[lrn]==tex.actualClass)
                foldN[tex.iterationNumber] += 1
        else:
            for tex in res.results:
                for lrn in range(res.numberOfLearners):
                    CAsByFold[lrn][tex.iterationNumber] += (tex.classes[lrn]==tex.actualClass) and tex.weight
                foldN[tex.iterationNumber] += tex.weight

        return statisticsByFolds(CAsByFold, foldN, reportSE, False)


# Obsolete, but kept for compatibility
def CA_se(res, **argkw):
    return CA(res, True, **argkw)


def AP(res, reportSE = False, **argkw):
    if res.numberOfIterations == 1:
        APs=[0.0]*res.numberOfLearners
        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
                APs = map(lambda res, probs: res + probs[tex.actualClass], APs, tex.probabilities)
            totweight = gettotsize(res)
        else:
            totweight = 0.
            for tex in res.results:
                APs = map(lambda res, probs: res + probs[tex.actualClass]*tex.weight, APs, tex.probabilities)
                totweight += tex.weight
        checkNonZero(totweight)
        return [AP/totweight for AP in APs]

    APsByFold = [[0.0]*res.numberOfLearners for i in range(res.numberOfIterations)]
    foldN = [0.0] * res.numberOfIterations
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            APsByFold[tex.iterationNumber] = map(lambda res, probs: res + probs[tex.actualClass], APsByFold[tex.iterationNumber], tex.probabilities)
            foldN[tex.iterationNumber] += 1
    else:
        for tex in res.results:
            APsByFold[tex.iterationNumber] = map(lambda res, probs: res + probs[tex.actualClass] * tex.weight, APsByFold[tex.iterationNumber], tex.probabilities)
            foldN[tex.iterationNumber] += tex.weight

    return statisticsByFolds(APsByFold, foldN, reportSE, True)


def BrierScore(res, reportSE = False, **argkw):
    """Computes Brier score"""
    # Computes an average (over examples) of sum_x(t(x) - p(x))^2, where
    #    x is class,
    #    t(x) is 0 for 'wrong' and 1 for 'correct' class
    #    p(x) is predicted probabilty.
    # There's a trick: since t(x) is zero for all classes but the
    # correct one (c), we compute the sum as sum_x(p(x)^2) - 2*p(c) + 1
    # Since +1 is there for each example, it adds 1 to the average
    # We skip the +1 inside the sum and add it just at the end of the function
    # We take max(result, 0) to avoid -0.0000x due to rounding errors

    if res.numberOfIterations == 1:
        MSEs=[0.0]*res.numberOfLearners
        if argkw.get("unweighted", 0) or not res.weights:
            totweight = 0.0
            for tex in res.results:
                MSEs = map(lambda res, probs:
                           res + reduce(lambda s, pi: s+pi**2, probs, 0) - 2*probs[tex.actualClass], MSEs, tex.probabilities)
                totweight += tex.weight
        else:
            for tex in res.results:
                MSEs = map(lambda res, probs:
                           res + tex.weight*reduce(lambda s, pi: s+pi**2, probs, 0) - 2*probs[tex.actualClass], MSEs, tex.probabilities)
            totweight = gettotweight(res)
        checkNonZero(totweight)
        if reportSE:
            return [(max(x/totweight+1.0, 0), 0) for x in MSEs]  ## change this, not zero!!!
        else:
            return [max(x/totweight+1.0, 0) for x in MSEs]

    BSs = [[0.0]*res.numberOfLearners for i in range(res.numberOfIterations)]
    foldN = [0.] * res.numberOfIterations

    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            BSs[tex.iterationNumber] = map(lambda rr, probs:
                       rr + reduce(lambda s, pi: s+pi**2, probs, 0) - 2*probs[tex.actualClass], BSs[tex.iterationNumber], tex.probabilities)
            foldN[tex.iterationNumber] += 1
    else:
        for tex in res.results:
            BSs[tex.iterationNumber] = map(lambda res, probs:
                       res + tex.weight*reduce(lambda s, pi: s+pi**2, probs, 0) - 2*probs[tex.actualClass], BSs[tex.iterationNumber], tex.probabilities)
            foldN[tex.iterationNumber] += tex.weight

    stats = statisticsByFolds(BSs, foldN, reportSE, True)
    if reportSE:
        return [(x+1.0, y) for x, y in stats]
    else:
        return [x+1.0 for x in stats]

def BSS(res, **argkw):
    return [1-x/2 for x in apply(BrierScore, (res, ), argkw)]

##def _KL_div(actualClass, predicted):
##    
##def KL(res, **argkw):
##    KLs = [0.0]*res.numberOfLearners
##
##    if argkw.get("unweighted", 0) or not res.weights:
##        for tex in res.results:
##            KLs = map(lambda res, predicted: res+KL(tex.actualClass, predicted), KLs, tex.probabilities)
##        totweight = gettotsize(res)
##    else:
##        for tex in res.results:
##            ## TEGA SE NISI!
##            CAs = map(lambda res, cls: res+(cls==tex.actualClass and tex.weight), CAs, tex.classes)
##        totweight = gettotweight(res)
##
##    return [x/totweight for x in CAs]

    
##def KL_se(res, **argkw):
##    # Kullback-Leibler divergence
##    if res.numberOfIterations==1:
##        if argkw.get("unweighted", 0) or not res.weights:
##            totweight = gettotsize(res)
##        else:
##            totweight = gettotweight(res)
##        return [(x, x*(1-x)/math.sqrt(totweight)) for x in apply(CA, (res,), argkw)]
##    else:
##        KLsByFold = [[0.0]*res.numberOfIterations for i in range(res.numberOfLearners)]
##        foldN = [0.0]*res.numberOfIterations
##
##        if argkw.get("unweighted", 0) or not res.weights:
##            for tex in res.results:
##                for lrn in range(res.numberOfLearners):
##                    CAsByFold[lrn][tex.iterationNumber] += 
##                foldN[tex.iterationNumber] += 1
##        else:
##            for tex in res.results:
##                for lrn in range(res.numberOfLearners):
##                    CAsByFold[lrn][tex.iterationNumber] += 
##                foldN[tex.iterationNumber] += tex.weight
##
##        newFolds = []
##        for lrn in range(res.numberOfLearners):
##            newF = []
##            for fold in range(res.numberOfIterations):
##                if foldN[fold]>0.0:
##                        newF.append(CAsByFold[lrn][fold]/foldN[fold])
##            newFolds.append(newF)
##
##        checkNonZero(len(newFolds))
##        return [(statc.mean(cas), statc.sterr(cas)) for cas in newFolds]
##

def IS_ex(Pc, P):
    "Pc aposterior probability, P aprior"
    if (Pc>=P):
        return -log2(P)+log2(Pc)
    else:
        return -(-log2(1-P)+log2(1-Pc))
    
def IS(res, apriori=None, reportSE = False, **argkw):
    if not apriori:
        apriori = classProbabilitiesFromRes(res)

    if res.numberOfIterations==1:
        ISs = [0.0]*res.numberOfLearners
        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
              for i in range(len(tex.probabilities)):
                    cls = tex.actualClass
                    ISs[i] += IS_ex(tex.probabilities[i][cls], apriori[cls])
            totweight = gettotsize(res)
        else:
            for tex in res.results:
              for i in range(len(tex.probabilities)):
                    cls = tex.actualClass
                    ISs[i] += IS_ex(tex.probabilities[i][cls], apriori[cls]) * tex.weight
            totweight = gettotweight(res)
        if reportSE:
            return [(IS/totweight,0) for IS in ISs]
        else:
            return [IS/totweight for IS in ISs]

        
    ISs = [[0.0]*res.numberOfIterations for i in range(res.numberOfLearners)]
    foldN = [0.] * res.numberOfIterations

    # compute info scores for each fold    
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            for i in range(len(tex.probabilities)):
                cls = tex.actualClass
                ISs[i][tex.iterationNumber] += IS_ex(tex.probabilities[i][cls], apriori[cls])
            foldN[tex.iterationNumber] += 1
    else:
        for tex in res.results:
            for i in range(len(tex.probabilities)):
                cls = tex.actualClass
                ISs[i][tex.iterationNumber] += IS_ex(tex.probabilities[i][cls], apriori[cls]) * tex.weight
            foldN[tex.iterationNumber] += tex.weight

    return statisticsByFolds(ISs, foldN, reportSE, False)


def Friedman(res, statistics, **argkw):
    sums = None
    for ri in splitByIterations(res):
        ranks = statc.rankdata(apply(statistics, (ri,), argkw))
        if sums:
            sums = sums and [ranks[i]+sums[i] for i in range(k)]
        else:
            sums = ranks
            k = len(sums)
    N = res.numberOfIterations
    k = len(sums)
    T = sum([x*x for x in sums])
    F = 12.0 / (N*k*(k+1)) * T  - 3 * N * (k+1)
    return F, statc.chisqprob(F, k-1)
    

def Wilcoxon(res, statistics, **argkw):
    res1, res2 = [], []
    for ri in splitByIterations(res):
        stats = apply(statistics, (ri,), argkw)
        if (len(stats) != 2):
            raise TypeError, "Wilcoxon compares two classifiers, no more, no less"
        res1.append(stats[0])
        res2.append(stats[1])
    return statc.wilcoxont(res1, res2)

def rankDifference(res, statistics, **argkw):
    if not res.results:
        raise TypeError, "no experiments"

    k = len(res.results[0].classes)
    if (k<2):
        raise TypeError, "nothing to compare (less than two classifiers given)"
    if k==2:
        return apply(Wilcoxon, (res, statistics), argkw)
    else:
        return apply(Friedman, (res, statistics), argkw)
    
class ConfusionMatrix:
    def __init__(self):
        self.TP = self.FN = self.FP = self.TN = 0.0

    def addTFPosNeg(self, predictedPositive, isPositive, weight = 1.0):
        if predictedPositive:
            if isPositive:
                self.TP += weight
            else:
                self.FP += weight
        else:
            if isPositive:
                self.FN += weight
            else:
                self.TN += weight


def confusionMatrices(res, classIndex=-1, **argkw):
    tfpns = [ConfusionMatrix() for i in range(res.numberOfLearners)]
    
    if classIndex<0:
        numberOfClasses = len(res.classValues)
        if classIndex < -1 or numberOfClasses > 2:
            cm = [[[0.0] * numberOfClasses for i in range(numberOfClasses)] for l in range(res.numberOfLearners)]
            if argkw.get("unweighted", 0) or not res.weights:
                for tex in res.results:
                    trueClass = int(tex.actualClass)
                    for li, pred in enumerate(tex.classes):
                        predClass = int(pred)
                        if predClass < numberOfClasses:
                            cm[li][trueClass][predClass] += 1
            else:
                for tex in enumerate(res.results):
                    trueClass = int(tex.actualClass)
                    for li, pred in tex.classes:
                        predClass = int(pred)
                        if predClass < numberOfClasses:
                            cm[li][trueClass][predClass] += tex.weight
            return cm
            
        elif res.baseClass>=0:
            classIndex = res.baseClass
        else:
            classIndex = 1
            
    cutoff = argkw.get("cutoff")
    if cutoff:
        if argkw.get("unweighted", 0) or not res.weights:
            for lr in res.results:
                isPositive=(lr.actualClass==classIndex)
                for i in range(res.numberOfLearners):
                    tfpns[i].addTFPosNeg(lr.probabilities[i][classIndex]>cutoff, isPositive)
        else:
            for lr in res.results:
                isPositive=(lr.actualClass==classIndex)
                for i in range(res.numberOfLearners):
                    tfpns[i].addTFPosNeg(lr.probabilities[i][classIndex]>cutoff, isPositive, lr.weight)
    else:
        if argkw.get("unweighted", 0) or not res.weights:
            for lr in res.results:
                isPositive=(lr.actualClass==classIndex)
                for i in range(res.numberOfLearners):
                    tfpns[i].addTFPosNeg(lr.classes[i]==classIndex, isPositive)
        else:
            for lr in res.results:
                isPositive=(lr.actualClass==classIndex)
                for i in range(res.numberOfLearners):
                    tfpns[i].addTFPosNeg(lr.classes[i]==classIndex, isPositive, lr.weight)
    return tfpns


# obsolete (renamed)
computeConfusionMatrices = confusionMatrices


def confusionChiSquare(confusionMatrix):
    dim = len(confusionMatrix)
    rowPriors = [sum(r) for r in confusionMatrix]
    colPriors = [sum([r[i] for r in confusionMatrix]) for i in range(dim)]
    total = sum(rowPriors)
    rowPriors = [r/total for r in rowPriors]
    colPriors = [r/total for r in colPriors]
    ss = 0
    for ri, row in enumerate(confusionMatrix):
        for ci, o in enumerate(row):
            e = total * rowPriors[ri] * colPriors[ci]
            if not e:
                return -1, -1, -1
            ss += (o-e)**2 / e
    df = (dim-1)**2
    return ss, df, statc.chisqprob(ss, df)
        
    
def sens(confm):
    """Return sensitivity (recall rate) over the given confusion matrix."""
    if type(confm) == list:
        return [sens(cm) for cm in confm]
    else:
        tot = confm.TP+confm.FN
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute sensitivity: one or both classes have no instances")
            return -1

        return confm.TP/tot

def recall(confm):
    """Return recall rate (sensitivity) over the given confusion matrix."""
    return sens(confm)


def spec(confm):
    """Return specificity over the given confusion matrix."""
    if type(confm) == list:
        return [spec(cm) for cm in confm]
    else:
        tot = confm.FP+confm.TN
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute specificity: one or both classes have no instances")
            return -1
        return confm.TN/tot
  

def PPV(confm):
    """Return positive predictive value (precision rate) over the given confusion matrix."""
    if type(confm) == list:
        return [PPV(cm) for cm in confm]
    else:
        tot = confm.TP+confm.FP
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute PPV: one or both classes have no instances")
            return -1
        return confm.TP/tot


def precision(confm):
    """Return precision rate (positive predictive value) over the given confusion matrix."""
    return PPV(confm)


def NPV(confm):
    """Return negative predictive value over the given confusion matrix."""
    if type(confm) == list:
        return [NPV(cm) for cm in confm]
    else:
        tot = confm.TP+confm.TN
        if tot < 1e-6:
            import warnings
            warnings.warn("Can't compute NPV: one or both classes have no instances")
            return -1
        return confm.TP/tot

def F1(confm):
    """Return F1 score (harmonic mean of precision and recall) over the given confusion matrix."""
    if type(confm) == list:
        return [F1(cm) for cm in confm]
    else:
        p = precision(confm)
        r = recall(confm)
        if p + r > 0:
            return 2. * p * r / (p + r)
        else:
            import warnings
            warnings.warn("Can't compute F1: P + R is zero or not defined")
            return -1

def Falpha(confm, alpha=1.0):
    """Return the alpha-mean of precision and recall over the given confusion matrix."""
    if type(confm) == list:
        return [Falpha(cm, alpha=alpha) for cm in confm]
    else:
        p = precision(confm)
        r = recall(confm)
        return (1. + alpha) * p * r / (alpha * p + r)
    
def MCC(confm):
    '''
    Return Mattew correlation coefficient over the given confusion matrix.

    MCC is calculated as follows:
    MCC = (TP*TN - FP*FN) / sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )
    
    [1] Matthews, B.W., Comparison of the predicted and observed secondary 
    structure of T4 phage lysozyme. Biochim. Biophys. Acta 1975, 405, 442-451

    code by Boris Gorelik
    '''
    if type(confm) == list:
        return [MCC(cm) for cm in confm]
    else:
        truePositive = confm.TP
        trueNegative = confm.TN
        falsePositive = confm.FP
        falseNegative = confm.FN 
          
        try:   
            r = (((truePositive * trueNegative) - (falsePositive * falseNegative))/ 
                math.sqrt(  (truePositive + falsePositive)  * 
                ( truePositive + falseNegative ) * 
                ( trueNegative + falsePositive ) * 
                ( trueNegative + falseNegative ) )
                )
        except ZeroDivisionError:
            # Zero difision occurs when there is either no true positives 
            # or no true negatives i.e. the problem contains only one 
            # type of classes. 
            r = None

    return r

def AUCWilcoxon(res, classIndex=-1, **argkw):
    import corn
    useweights = res.weights and not argkw.get("unweighted", 0)
    problists, tots = corn.computeROCCumulative(res, classIndex, useweights)

    results=[]

    totPos, totNeg = tots[1], tots[0]
    N = totPos + totNeg
    for plist in problists:
        highPos, lowNeg = totPos, 0.0
        W, Q1, Q2 = 0.0, 0.0, 0.0
        for prob in plist:
            thisPos, thisNeg = prob[1][1], prob[1][0]
            highPos -= thisPos
            W += thisNeg * (highPos + thisPos/2.)
            Q2 += thisPos * (lowNeg**2  + lowNeg*thisNeg  + thisNeg**2 /3.)
            Q1 += thisNeg * (highPos**2 + highPos*thisPos + thisPos**2 /3.)

            lowNeg += thisNeg

        W  /= (totPos*totNeg)
        Q1 /= (totNeg*totPos**2)
        Q2 /= (totPos*totNeg**2)

        SE = math.sqrt( (W*(1-W) + (totPos-1)*(Q1-W**2) + (totNeg-1)*(Q2-W**2)) / (totPos*totNeg) )
        results.append((W, SE))
    return results

AROC = AUCWilcoxon # for backward compatibility, AROC is obsolote

def compare2AUCs(res, lrn1, lrn2, classIndex=-1, **argkw):
    import corn
    return corn.compare2ROCs(res, lrn1, lrn2, classIndex, res.weights and not argkw.get("unweighted"))

compare2AROCs = compare2AUCs # for backward compatibility, compare2AROCs is obsolote

    
def computeROC(res, classIndex=-1):
    import corn
    problists, tots = corn.computeROCCumulative(res, classIndex)

    results = []
    totPos, totNeg = tots[1], tots[0]

    for plist in problists:
        curve=[(1., 1.)]
        TP, TN = totPos, 0.0
        FN, FP = 0., totNeg
        for prob in plist:
            thisPos, thisNeg = prob[1][1], prob[1][0]
            # thisPos go from TP to FN
            TP -= thisPos
            FN += thisPos
            # thisNeg go from FP to TN
            TN += thisNeg
            FP -= thisNeg

            sens = TP/(TP+FN)
            spec = TN/(FP+TN)
            curve.append((1-spec, sens))
        results.append(curve)

    return results    

## TC's implementation of algorithms, taken from:
## T Fawcett: ROC Graphs: Notes and Practical Considerations for Data Mining Researchers, submitted to KDD Journal. 
def ROCslope((P1x, P1y, P1fscore), (P2x, P2y, P2fscore)):
    if (P1x == P2x):
        return 1e300
    return (P1y - P2y) / (P1x - P2x)

def ROCaddPoint(P, R, keepConcavities=1):
    if keepConcavities:
        R.append(P)
    else:
        while (1):
            if len(R) < 2:
                R.append(P)
                return R
            else:
                T = R.pop()
                T2 = R[-1]
                if ROCslope(T2, T) > ROCslope(T, P):
                    R.append(T)
                    R.append(P)
                    return R
    return R

def TCcomputeROC(res, classIndex=-1, keepConcavities=1):
    import corn
    problists, tots = corn.computeROCCumulative(res, classIndex)

    results = []
    P, N = tots[1], tots[0]

    for plist in problists:
        ## corn gives an increasing by scores list, we need a decreasing by scores
        plist.reverse()
        TP = 0.0
        FP = 0.0
        curve=[]
        fPrev = 10e300 # "infinity" score at 0.0, 0.0
        for prob in plist:
            f = prob[0]
            if f <> fPrev:
                if P:
                    tpr = TP/P
                else:
                    tpr = 0.0
                if N:
                    fpr = FP/N
                else:
                    fpr = 0.0
                curve = ROCaddPoint((fpr, tpr, fPrev), curve, keepConcavities)
                fPrev = f
            thisPos, thisNeg = prob[1][1], prob[1][0]
            TP += thisPos
            FP += thisNeg
        if P:
            tpr = TP/P
        else:
            tpr = 0.0
        if N:
            fpr = FP/N
        else:
            fpr = 0.0
        curve = ROCaddPoint((fpr, tpr, f), curve, keepConcavities) ## ugly
        results.append(curve)

    return results

## returns a list of points at the intersection of the tangential iso-performance line and the given ROC curve
## for given values of FPcost, FNcost and pval
def TCbestThresholdsOnROCcurve(FPcost, FNcost, pval, curve):
    m = (FPcost*(1.0 - pval)) / (FNcost*pval)

    ## put the iso-performance line in point (0.0, 1.0)
    x0, y0 = (0.0, 1.0)
    x1, y1 = (1.0, 1.0 + m)
    d01 = math.sqrt((x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0))

    ## calculate and find the closest point to the line
    firstp = 1
    mind = 0.0
    a = (x0*y1 - x1*y0)
    closestPoints = []
    for (x, y, fscore) in curve:
        d = ((y0 - y1)*x + (x1 - x0)*y + a) / d01
        d = abs(d)
        if firstp or d < mind:
            mind, firstp = d, 0
            closestPoints = [(x, y, fscore)]
        else:
            if abs(d - mind) <= 0.0001: ## close enough
                closestPoints.append( (x, y, fscore) )
    return closestPoints          

def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."

    if end == None:
        end = start + 0.0
        start = 0.0

    if inc == None or inc == 0:
        inc = 1.0

    L = [start]
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            L.append(end)
            break
        elif inc < 0 and next <= end:
            L.append(end)
            break
        L.append(next)
        
    return L

## input ROCcurves are of form [ROCcurves1, ROCcurves2, ... ROCcurvesN],
## where ROCcurvesX is a set of ROC curves,
## where a (one) ROC curve is a set of (FP, TP) points
##
## for each (sub)set of input ROC curves
## returns the average ROC curve and an array of (vertical) standard deviations
def TCverticalAverageROC(ROCcurves, samples = 10):
    def INTERPOLATE((P1x, P1y, P1fscore), (P2x, P2y, P2fscore), X):
        if (P1x == P2x) or ((X > P1x) and (X > P2x)) or ((X < P1x) and (X < P2x)):
            raise SystemError, "assumptions for interpolation are not met: P1 = %f,%f P2 = %f,%f X = %f" % (P1x, P1y, P2x, P2y, X)
        dx = float(P2x) - float(P1x)
        dy = float(P2y) - float(P1y)
        m = dy/dx
        return P1y + m*(X - P1x)

    def TP_FOR_FP(FPsample, ROC, npts):
        i = 0
        while i < npts - 1:
            (fp, _, _) = ROC[i + 1]
            if (fp <= FPsample):
                i += 1
            else:
                break
        (fp, tp, _) = ROC[i]
        if fp == FPsample:
            return tp
        elif fp < FPsample:
            return INTERPOLATE(ROC[i], ROC[i+1], FPsample)
        raise SystemError, "cannot compute: TP_FOR_FP in TCverticalAverageROC"
        #return 0.0

    average = []
    stdev = []
    for ROCS in ROCcurves:
        npts = []
        for c in ROCS:
            npts.append(len(c))
        nrocs = len(ROCS)

        TPavg = []
        TPstd = []
        for FPsample in frange(0.0, 1.0, 1.0/samples):
            TPsum = []
            for i in range(nrocs):
                TPsum.append( TP_FOR_FP(FPsample, ROCS[i], npts[i]) ) ##TPsum = TPsum + TP_FOR_FP(FPsample, ROCS[i], npts[i])
            TPavg.append( (FPsample, statc.mean(TPsum)) )
            if len(TPsum) > 1:
                stdv = statc.std(TPsum)
            else:
                stdv = 0.0
            TPstd.append( stdv )

        average.append(TPavg)
        stdev.append(TPstd)

    return (average, stdev)

## input ROCcurves are of form [ROCcurves1, ROCcurves2, ... ROCcurvesN],
## where ROCcurvesX is a set of ROC curves,
## where a (one) ROC curve is a set of (FP, TP) points
##
## for each (sub)set of input ROC curves
## returns the average ROC curve, an array of vertical standard deviations and an array of horizontal standard deviations
def TCthresholdlAverageROC(ROCcurves, samples = 10):
    def POINT_AT_THRESH(ROC, npts, thresh):
        i = 0
        while i < npts - 1:
            (px, py, pfscore) = ROC[i]
            if (pfscore > thresh):
                i += 1
            else:
                break
        return ROC[i]

    average = []
    stdevV = []
    stdevH = []
    for ROCS in ROCcurves:
        npts = []
        for c in ROCS:
            npts.append(len(c))
        nrocs = len(ROCS)

        T = []
        for c in ROCS:
            for (px, py, pfscore) in c:
##                try:
##                    T.index(pfscore)
##                except:
                T.append(pfscore)
        T.sort()
        T.reverse() ## ugly

        TPavg = []
        TPstdV = []
        TPstdH = []
        for tidx in frange(0, (len(T) - 1.0), float(len(T))/samples):
            FPsum = []
            TPsum = []
            for i in range(nrocs):
                (fp, tp, _) = POINT_AT_THRESH(ROCS[i], npts[i], T[int(tidx)])
                FPsum.append(fp)
                TPsum.append(tp)
            TPavg.append( (statc.mean(FPsum), statc.mean(TPsum)) )
            ## vertical standard deviation
            if len(TPsum) > 1:
                stdv = statc.std(TPsum)
            else:
                stdv = 0.0
            TPstdV.append( stdv )
            ## horizontal standard deviation
            if len(FPsum) > 1:
                stdh = statc.std(FPsum)
            else:
                stdh = 0.0
            TPstdH.append( stdh )

        average.append(TPavg)
        stdevV.append(TPstdV)
        stdevH.append(TPstdH)

    return (average, stdevV, stdevH)

## Calibration Curve
## returns an array of (curve, yesClassPredictions, noClassPredictions) elements, where:
##  - curve is an array of points (x, y) on the calibration curve
##  - yesClassRugPoints is an array of (x, 1) points
##  - noClassRugPoints is an array of (x, 0) points
def computeCalibrationCurve(res, classIndex=-1):
    import corn
    ## merge multiple iterations into one
    mres = orngTest.ExperimentResults(1, res.classifierNames, res.classValues, res.weights, classifiers=res.classifiers, loaded=res.loaded)
    for te in res.results:
        mres.results.append( te )

    problists, tots = corn.computeROCCumulative(mres, classIndex)

    results = []
    P, N = tots[1], tots[0]

    bins = 10 ## divide interval between 0.0 and 1.0 into N bins

    for plist in problists:
        yesClassRugPoints = [] 
        noClassRugPoints = []

        yesBinsVals = [0] * bins
        noBinsVals = [0] * bins
        for (f, (thisNeg, thisPos)) in plist:
            yesClassRugPoints.append( (f, thisPos) ) #1.0
            noClassRugPoints.append( (f, thisNeg) ) #1.0

            index = int(f * bins )
            index = min(index, bins - 1) ## just in case for value 1.0
            yesBinsVals[index] += thisPos
            noBinsVals[index] += thisNeg

        curve = []
        for cn in range(bins):
            f = float(cn * 1.0 / bins) + (1.0 / 2.0 / bins)
            yesVal = yesBinsVals[cn]
            noVal = noBinsVals[cn]
            allVal = yesVal + noVal
            if allVal == 0.0: continue
            y = float(yesVal)/float(allVal)
            curve.append( (f,  y) )

        ## smooth the curve
        maxnPoints = 100
        if len(curve) >= 3:
#            loessCurve = statc.loess(curve, -3, 0.6)
            loessCurve = statc.loess(curve, maxnPoints, 0.5, 3)
        else:
            loessCurve = curve
        clen = len(loessCurve)
        if clen > maxnPoints:
            df = clen / maxnPoints
            if df < 1: df = 1
            curve = [loessCurve[i]  for i in range(0, clen, df)]
        else:
            curve = loessCurve
        curve = [(c)[:2] for c in curve] ## remove the third value (variance of epsilon?) that suddenly appeared in the output of the statc.loess function
        results.append((curve, yesClassRugPoints, noClassRugPoints))

    return results


## Lift Curve
## returns an array of curve elements, where:
##  - curve is an array of points ((TP+FP)/(P + N), TP/P, (th, FP/N)) on the Lift Curve
def computeLiftCurve(res, classIndex=-1):
    import corn
    ## merge multiple iterations into one
    mres = orngTest.ExperimentResults(1, res.classifierNames, res.classValues, res.weights, classifiers=res.classifiers, loaded=res.loaded)
    for te in res.results:
        mres.results.append( te )

    problists, tots = corn.computeROCCumulative(mres, classIndex)

    results = []
    P, N = tots[1], tots[0]

    for plist in problists:
        ## corn gives an increasing by scores list, we need a decreasing by scores
        plist.reverse()
        TP = 0.0
        FP = 0.0
        curve = [(0.0, 0.0, (10e300, 0.0))]
        for (f, (thisNeg, thisPos)) in plist:
            TP += thisPos
            FP += thisNeg
            curve.append( ((TP+FP)/(P + N), TP, (f, FP/N)) )
        results.append(curve)

    return P, N, results
###

class CDT:
  """ Stores number of concordant (C), discordant (D) and tied (T) pairs (used for AUC) """
  def __init__(self, C=0.0, D=0.0, T=0.0):
    self.C, self.D, self.T = C, D, T
   
def isCDTEmpty(cdt):
    return cdt.C + cdt.D + cdt.T < 1e-20


def computeCDT(res, classIndex=-1, **argkw):
    """Obsolete, don't use"""
    import corn
    if classIndex<0:
        if res.baseClass>=0:
            classIndex = res.baseClass
        else:
            classIndex = 1
            
    useweights = res.weights and not argkw.get("unweighted", 0)
    weightByClasses = argkw.get("weightByClasses", True)

    if (res.numberOfIterations>1):
        CDTs = [CDT() for i in range(res.numberOfLearners)]
        iterationExperiments = splitByIterations(res)
        for exp in iterationExperiments:
            expCDTs = corn.computeCDT(exp, classIndex, useweights)
            for i in range(len(CDTs)):
                CDTs[i].C += expCDTs[i].C
                CDTs[i].D += expCDTs[i].D
                CDTs[i].T += expCDTs[i].T
        for i in range(res.numberOfLearners):
            if isCDTEmpty(CDTs[0]):
                return corn.computeCDT(res, classIndex, useweights)
        
        return CDTs
    else:
        return corn.computeCDT(res, classIndex, useweights)

## THIS FUNCTION IS OBSOLETE AND ITS AVERAGING OVER FOLDS IS QUESTIONABLE
## DON'T USE IT
def ROCsFromCDT(cdt, **argkw):
    """Obsolete, don't use"""
    if type(cdt) == list:
        return [ROCsFromCDT(c) for c in cdt]

    C, D, T = cdt.C, cdt.D, cdt.T
    N = C+D+T
    if N < 1e-6:
        import warnings
        warnings.warn("Can't compute AUC: one or both classes have no instances")
        return (-1,)*8
    if N < 2:
        import warnings
        warnings.warn("Can't compute AUC: one or both classes have too few examples")

    som = (C-D)/N
    c = 0.5*(1+som)
  
    if (C+D):
        res = (C/N*100, D/N*100, T/N*100, N, som, (C-D)/(C+D), (C-D)/(N*(N-1)/2), 0.5*(1+som))
    else:
        res = (C/N*100, D/N*100, T/N*100, N, som, -1.0, (C-D)/(N*(N-1)/2), 0.5*(1+som))

    if argkw.get("print"):
        print "Concordant  = %5.1f       Somers' D = %1.3f" % (res[0], res[4])
        print "Discordant  = %5.1f       Gamma     = %1.3f" % (res[1], res[5]>0 and res[5] or "N/A")
        print "Tied        = %5.1f       Tau-a     = %1.3f" % (res[2], res[6])
        print " %6d pairs             c         = %1.3f"    % (res[3], res[7])

    return res

AROCFromCDT = ROCsFromCDT  # for backward compatibility, AROCFromCDT is obsolote



# computes AUC using a specified 'cdtComputer' function
# It tries to compute AUCs from 'ite' (examples from a single iteration) and,
# if C+D+T=0, from 'all_ite' (entire test set). In the former case, the AUCs
# are divided by 'divideByIfIte'. Additional flag is returned which is True in
# the former case, or False in the latter.
def AUC_x(cdtComputer, ite, all_ite, divideByIfIte, computerArgs):
    cdts = cdtComputer(*(ite, ) + computerArgs)
    if not isCDTEmpty(cdts[0]):
        return [(cdt.C+cdt.T/2)/(cdt.C+cdt.D+cdt.T)/divideByIfIte for cdt in cdts], True
        
    if all_ite:
         cdts = cdtComputer(*(all_ite, ) + computerArgs)
         if not isCDTEmpty(cdts[0]):
             return [(cdt.C+cdt.T/2)/(cdt.C+cdt.D+cdt.T) for cdt in cdts], False

    return False, False

    
# computes AUC between classes i and j as if there we no other classes
def AUC_ij(ite, classIndex1, classIndex2, useWeights = True, all_ite = None, divideByIfIte = 1.0):
    import corn
    return AUC_x(corn.computeCDTPair, ite, all_ite, divideByIfIte, (classIndex1, classIndex2, useWeights))


# computes AUC between class i and the other classes (treating them as the same class)
def AUC_i(ite, classIndex, useWeights = True, all_ite = None, divideByIfIte = 1.0):
    import corn
    return AUC_x(corn.computeCDT, ite, all_ite, divideByIfIte, (classIndex, useWeights))
   

# computes the average AUC over folds using a "AUCcomputer" (AUC_i or AUC_ij)
# it returns the sum of what is returned by the computer, unless at a certain
# fold the computer has to resort to computing over all folds or even this failed;
# in these cases the result is returned immediately
def AUC_iterations(AUCcomputer, iterations, computerArgs):
    subsum_aucs = [0.] * iterations[0].numberOfLearners
    for ite in iterations:
        aucs, foldsUsed = AUCcomputer(*(ite, ) + computerArgs)
        if not aucs:
            return None
        if not foldsUsed:
            return aucs
        subsum_aucs = map(add, subsum_aucs, aucs)
    return subsum_aucs


# AUC for binary classification problems
def AUC_binary(res, useWeights = True):
    if res.numberOfIterations > 1:
        return AUC_iterations(AUC_i, splitByIterations(res), (-1, useWeights, res, res.numberOfIterations))
    else:
        return AUC_i(res, -1, useWeights)[0]

# AUC for multiclass problems
def AUC_multi(res, useWeights = True, method = 0):
    numberOfClasses = len(res.classValues)
    
    if res.numberOfIterations > 1:
        iterations = splitByIterations(res)
        all_ite = res
    else:
        iterations = [res]
        all_ite = None
    
    # by pairs
    sum_aucs = [0.] * res.numberOfLearners
    usefulClassPairs = 0.

    if method in [0, 2]:
        prob = classProbabilitiesFromRes(res)
        
    if method <= 1:
        for classIndex1 in range(numberOfClasses):
            for classIndex2 in range(classIndex1):
                subsum_aucs = AUC_iterations(AUC_ij, iterations, (classIndex1, classIndex2, useWeights, all_ite, res.numberOfIterations))
                if subsum_aucs:
                    if method == 0:
                        p_ij = prob[classIndex1] * prob[classIndex2]
                        subsum_aucs = [x * p_ij  for x in subsum_aucs]
                        usefulClassPairs += p_ij
                    else:
                        usefulClassPairs += 1
                    sum_aucs = map(add, sum_aucs, subsum_aucs)
    else:
        for classIndex in range(numberOfClasses):
            subsum_aucs = AUC_iterations(AUC_i, iterations, (classIndex, useWeights, all_ite, res.numberOfIterations))
            if subsum_aucs:
                if method == 0:
                    p_i = prob[classIndex]
                    subsum_aucs = [x * p_i  for x in subsum_aucs]
                    usefulClassPairs += p_i
                else:
                    usefulClassPairs += 1
                sum_aucs = map(add, sum_aucs, subsum_aucs)
                    
    if usefulClassPairs > 0:
        sum_aucs = [x/usefulClassPairs for x in sum_aucs]

    return sum_aucs


# Computes AUC, possibly for multiple classes (the averaging method can be specified)
# Results over folds are averages; if some folds examples from one class only, the folds are merged
def AUC(res, method = 0, useWeights = True):
    if len(res.classValues) < 2:
        raise "Cannot compute AUC on a single-class problem"
    elif len(res.classValues) == 2:
        return AUC_binary(res, useWeights)
    else:
        return AUC_multi(res, useWeights, method)

AUC.ByWeightedPairs = 0
AUC.ByPairs = 1
AUC.WeightedOneAgainstAll = 2
AUC.OneAgainstAll = 3


# Computes AUC; in multivalued class problem, AUC is computed as one against all
# Results over folds are averages; if some folds examples from one class only, the folds are merged
def AUC_single(res, classIndex = -1, useWeights = True):
    if classIndex<0:
        if res.baseClass>=0:
            classIndex = res.baseClass
        else:
            classIndex = 1

    if res.numberOfIterations > 1:
        return AUC_iterations(AUC_i, splitByIterations(res), (classIndex, useWeights, res, res.numberOfIterations))
    else:
        return AUC_i( res, classIndex, useWeights)[0]

# Computes AUC for a pair of classes (as if there were no other classes)
# Results over folds are averages; if some folds have examples from one class only, the folds are merged
def AUC_pair(res, classIndex1, classIndex2, useWeights = True):
    if res.numberOfIterations > 1:
        return AUC_iterations(AUC_ij, splitByIterations(res), (classIndex1, classIndex2, useWeights, res, res.numberOfIterations))
    else:
        return AUC_ij(res, classIndex1, classIndex2, useWeights)
  

# AUC for multiclass problems
def AUC_matrix(res, useWeights = True):
    numberOfClasses = len(res.classValues)
    numberOfLearners = res.numberOfLearners
    
    if res.numberOfIterations > 1:
        iterations, all_ite = splitByIterations(res), res
    else:
        iterations, all_ite = [res], None
    
    aucs = [[[] for i in range(numberOfClasses)] for i in range(numberOfLearners)]
    prob = classProbabilitiesFromRes(res)
        
    for classIndex1 in range(numberOfClasses):
        for classIndex2 in range(classIndex1):
            pair_aucs = AUC_iterations(AUC_ij, iterations, (classIndex1, classIndex2, useWeights, all_ite, res.numberOfIterations))
            if pair_aucs:
                for lrn in range(numberOfLearners):
                    aucs[lrn][classIndex1].append(pair_aucs[lrn])
            else:
                for lrn in range(numberOfLearners):
                    aucs[lrn][classIndex1].append(-1)
    return aucs
                

def McNemar(res, **argkw):
    nLearners = res.numberOfLearners
    mcm = []
    for i in range(nLearners):
       mcm.append([0.0]*res.numberOfLearners)

    if not res.weights or argkw.get("unweighted"):
        for i in res.results:
            actual = i.actualClass
            classes = i.classes
            for l1 in range(nLearners):
                for l2 in range(l1, nLearners):
                    if classes[l1]==actual:
                        if classes[l2]!=actual:
                            mcm[l1][l2] += 1
                    elif classes[l2]==actual:
                        mcm[l2][l1] += 1
    else:
        for i in res.results:
            actual = i.actualClass
            classes = i.classes
            for l1 in range(nLearners):
                for l2 in range(l1, nLearners):
                    if classes[l1]==actual:
                        if classes[l2]!=actual:
                            mcm[l1][l2] += i.weight
                    elif classes[l2]==actual:
                        mcm[l2][l1] += i.weight

    for l1 in range(nLearners):
        for l2 in range(l1, nLearners):
            su=mcm[l1][l2] + mcm[l2][l1]
            if su:
                mcm[l2][l1] = (abs(mcm[l1][l2]-mcm[l2][l1])-1)**2 / su
            else:
                mcm[l2][l1] = 0

    for l1 in range(nLearners):
        mcm[l1]=mcm[l1][:l1]

    return mcm


def McNemarOfTwo(res, lrn1, lrn2):
    tf = ft = 0.0
    if not res.weights or argkw.get("unweighted"):
        for i in res.results:
            actual=i.actualClass
            if i.classes[lrn1]==actual:
                if i.classes[lrn2]!=actual:
                    tf += i.weight
            elif i.classes[lrn2]==actual:
                    ft += i.weight
    else:
        for i in res.results:
            actual=i.actualClass
            if i.classes[lrn1]==actual:
                if i.classes[lrn2]!=actual:
                    tf += 1.0
            elif i.classes[lrn2]==actual:
                    ft += 1.0

    su = tf + ft
    if su:
        return (abs(tf-ft)-1)**2 / su
    else:
        return 0


def Friedman(res, stat=CA):
    """ Compares classifiers by Friedman test, treating folds as different examles.
        Returns F, p and average ranks
    """
    res_split = splitByIterations(res)
    res = [stat(r) for r in res_split]
    
    N = len(res)
    k = len(res[0])
    sums = [0.0]*k
    for r in res:
        ranks = [k-x+1 for x in statc.rankdata(r)]
        if stat==BrierScore: # reverse ranks for BrierScore (lower better)
            ranks = [k+1-x for x in ranks]
        sums = [ranks[i]+sums[i] for i in range(k)]

    T = sum([x*x for x in sums])
    sums = [x/N for x in sums]

    F = 12.0 / (N*k*(k+1)) * T  - 3 * N * (k+1)

    return F, statc.chisqprob(F, k-1), sums


def WilcoxonPairs(res, avgranks, stat=CA):
    """ Returns a triangular matrix, where element[i][j] stores significance of difference
        between i-th and j-th classifier, as computed by Wilcoxon test. The element is positive
        if i-th is better than j-th, negative if it is worse, and 1 if they are equal.
        Arguments to function are ExperimentResults, average ranks (as returned by Friedman)
        and, optionally, a statistics; greater values should mean better results.append
    """
    res_split = splitByIterations(res)
    res = [stat(r) for r in res_split]

    k = len(res[0])
    bt = []
    for m1 in range(k):
        nl = []
        for m2 in range(m1+1, k):
            t, p = statc.wilcoxont([r[m1] for r in res], [r[m2] for r in res])
            if avgranks[m1]<avgranks[m2]:
                nl.append(p)
            elif avgranks[m2]<avgranks[m1]:
                nl.append(-p)
            else:
                nl.append(1)
        bt.append(nl)
    return bt


def plotLearningCurveLearners(file, allResults, proportions, learners, noConfidence=0):
    plotLearningCurve(file, allResults, proportions, [orngMisc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))], noConfidence)
    
def plotLearningCurve(file, allResults, proportions, legend, noConfidence=0):
    import types
    fopened=0
    if (type(file)==types.StringType):
        file=open(file, "wt")
        fopened=1
        
    file.write("set yrange [0:1]\n")
    file.write("set xrange [%f:%f]\n" % (proportions[0], proportions[-1]))
    file.write("set multiplot\n\n")
    CAs = [CA_dev(x) for x in allResults]

    file.write("plot \\\n")
    for i in range(len(legend)-1):
        if not noConfidence:
            file.write("'-' title '' with yerrorbars pointtype %i,\\\n" % (i+1))
        file.write("'-' title '%s' with linespoints pointtype %i,\\\n" % (legend[i], i+1))
    if not noConfidence:
        file.write("'-' title '' with yerrorbars pointtype %i,\\\n" % (len(legend)))
    file.write("'-' title '%s' with linespoints pointtype %i\n" % (legend[-1], len(legend)))

    for i in range(len(legend)):
        if not noConfidence:
            for p in range(len(proportions)):
                file.write("%f\t%f\t%f\n" % (proportions[p], CAs[p][i][0], 1.96*CAs[p][i][1]))
            file.write("e\n\n")

        for p in range(len(proportions)):
            file.write("%f\t%f\n" % (proportions[p], CAs[p][i][0]))
        file.write("e\n\n")

    if fopened:
        file.close()


def printSingleROCCurveCoordinates(file, curve):
    import types
    fopened=0
    if (type(file)==types.StringType):
        file=open(file, "wt")
        fopened=1

    for coord in curve:
        file.write("%5.3f\t%5.3f\n" % tuple(coord))

    if fopened:
        file.close()


def plotROCLearners(file, curves, learners):
    plotROC(file, curves, [orngMisc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))])
    
def plotROC(file, curves, legend):
    import types
    fopened=0
    if (type(file)==types.StringType):
        file=open(file, "wt")
        fopened=1

    file.write("set yrange [0:1]\n")
    file.write("set xrange [0:1]\n")
    file.write("set multiplot\n\n")

    file.write("plot \\\n")
    for leg in legend:
        file.write("'-' title '%s' with lines,\\\n" % leg)
    file.write("'-' title '' with lines\n")

    for curve in curves:
        for coord in curve:
            file.write("%5.3f\t%5.3f\n" % tuple(coord))
        file.write("e\n\n")

    file.write("1.0\t1.0\n0.0\t0.0e\n\n")          

    if fopened:
        file.close()



def plotMcNemarCurveLearners(file, allResults, proportions, learners, reference=-1):
    plotMcNemarCurve(file, allResults, proportions, [orngMisc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))], reference)

def plotMcNemarCurve(file, allResults, proportions, legend, reference=-1):
    if reference<0:
        reference=len(legend)-1
        
    import types
    fopened=0
    if (type(file)==types.StringType):
        file=open(file, "wt")
        fopened=1
        
    #file.write("set yrange [0:1]\n")
    #file.write("set xrange [%f:%f]\n" % (proportions[0], proportions[-1]))
    file.write("set multiplot\n\n")
    file.write("plot \\\n")
    tmap=range(reference)+range(reference+1, len(legend))
    for i in tmap[:-1]:
        file.write("'-' title '%s' with linespoints pointtype %i,\\\n" % (legend[i], i+1))
    file.write("'-' title '%s' with linespoints pointtype %i\n" % (legend[tmap[-1]], tmap[-1]))
    file.write("\n")

    for i in tmap:
        for p in range(len(proportions)):
            file.write("%f\t%f\n" % (proportions[p], McNemarOfTwo(allResults[p], i, reference)))
        file.write("e\n\n")

    if fopened:
        file.close()

defaultPointTypes=("{$\\circ$}", "{$\\diamond$}", "{$+$}", "{$\\times$}", "{$|$}")+tuple([chr(x) for x in range(97, 122)])
defaultLineTypes=("\\setsolid", "\\setdashpattern <4pt, 2pt>", "\\setdashpattern <8pt, 2pt>", "\\setdashes", "\\setdots")

def learningCurveLearners2PiCTeX(file, allResults, proportions, **options):
    return apply(learningCurve2PiCTeX, (file, allResults, proportions), options)
    
def learningCurve2PiCTeX(file, allResults, proportions, **options):
    import types
    fopened=0
    if (type(file)==types.StringType):
        file=open(file, "wt")
        fopened=1

    nexamples=len(allResults[0].results)
    CAs = [CA_dev(x) for x in allResults]

    graphsize=float(options.get("graphsize", 10.0)) #cm
    difprop=proportions[-1]-proportions[0]
    ntestexamples=nexamples*proportions[-1]
    xunit=graphsize/ntestexamples

    yshift=float(options.get("yshift", -ntestexamples/20.))
    
    pointtypes=options.get("pointtypes", defaultPointTypes)
    linetypes=options.get("linetypes", defaultLineTypes)

    if options.has_key("numberedx"):
        numberedx=options["numberedx"]
        if type(numberedx)==types.IntType:
            if numberedx>0:
                numberedx=[nexamples*proportions[int(i/float(numberedx)*len(proportions))] for i in range(numberedx)]+[proportions[-1]*nexamples]
            elif numberedx<0:
                numberedx = -numberedx
                newn=[]
                for i in range(numberedx+1):
                    wanted=proportions[0]+float(i)/numberedx*difprop
                    best=(10, 0)
                    for t in proportions:
                        td=abs(wanted-t)
                        if td<best[0]:
                            best=(td, t)
                    if not best[1] in newn:
                        newn.append(best[1])
                newn.sort()
                numberedx=[nexamples*x for x in newn]
        elif type(numberedx[0])==types.FloatType:
            numberedx=[nexamples*x for x in numberedx]
    else:
        numberedx=[nexamples*x for x in proportions]

    file.write("\\mbox{\n")
    file.write("  \\beginpicture\n")
    file.write("  \\setcoordinatesystem units <%10.8fcm, %5.3fcm>\n\n" % (xunit, graphsize))    
    file.write("  \\setplotarea x from %5.3f to %5.3f, y from 0 to 1\n" % (0, ntestexamples))    
    file.write("  \\axis bottom invisible\n")# label {#examples}\n")
    file.write("      ticks short at %s /\n" % reduce(lambda x,y:x+" "+y, ["%i"%(x*nexamples+0.5) for x in proportions]))
    if numberedx:
        file.write("            long numbered at %s /\n" % reduce(lambda x,y:x+y, ["%i " % int(x+0.5) for x in numberedx]))
    file.write("  /\n")
    file.write("  \\axis left invisible\n")# label {classification accuracy}\n")
    file.write("      shiftedto y=%5.3f\n" % yshift)
    file.write("      ticks short from 0.0 to 1.0 by 0.05\n")
    file.write("            long numbered from 0.0 to 1.0 by 0.25\n")
    file.write("  /\n")
    if options.has_key("default"):
        file.write("  \\setdashpattern<1pt, 1pt>\n")
        file.write("  \\plot %5.3f %5.3f %5.3f %5.3f /\n" % (0., options["default"], ntestexamples, options["default"]))
    
    for i in range(len(CAs[0])):
        coordinates=reduce(lambda x,y:x+" "+y, ["%i %5.3f" % (proportions[p]*nexamples, CAs[p][i][0]) for p in range(len(proportions))])
        if linetypes:
            file.write("  %s\n" % linetypes[i])
            file.write("  \\plot %s /\n" % coordinates)
        if pointtypes:
            file.write("  \\multiput %s at %s /\n" % (pointtypes[i], coordinates))

    file.write("  \\endpicture\n")
    file.write("}\n")
    if fopened:
        file.close()
    file.close()
    del file

def legendLearners2PiCTeX(file, learners, **options):
  return apply(legend2PiCTeX, (file, [orngMisc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))]), options)
    
def legend2PiCTeX(file, legend, **options):
    import types
    fopened=0
    if (type(file)==types.StringType):
        file=open(file, "wt")
        fopened=1

    pointtypes=options.get("pointtypes", defaultPointTypes)
    linetypes=options.get("linetypes", defaultLineTypes)

    file.write("\\mbox{\n")
    file.write("  \\beginpicture\n")
    file.write("  \\setcoordinatesystem units <5cm, 1pt>\n\n")
    file.write("  \\setplotarea x from 0.000 to %5.3f, y from 0 to 12\n" % len(legend))

    for i in range(len(legend)):
        if linetypes:
            file.write("  %s\n" % linetypes[i])
            file.write("  \\plot %5.3f 6 %5.3f 6 /\n" % (i, i+0.2))
        if pointtypes:
            file.write("  \\put {%s} at %5.3f 6\n" % (pointtypes[i], i+0.1))
        file.write("  \\put {%s} [lb] at %5.3f 0\n" % (legend[i], i+0.25))

    file.write("  \\endpicture\n")
    file.write("}\n")
    if fopened:
        file.close()
    file.close()
    del file


def compute_friedman(avranks, N):
    """
    Returns a tuple (friedman statistic, degrees of freedom)
    and (Iman statistic - F-distribution, degrees of freedom)
    """

    k = len(avranks)

    def friedman(N, k, ranks):
        return 12*N*(sum([rank**2.0 for rank in ranks]) - (k*(k+1)*(k+1)/4.0) )/(k*(k+1))

    def iman(fried, N, k):
        return (N-1)*fried/(N*(k-1) - fried)

    f = friedman(N, k, avranks)
    im = iman(f, N, k)
    fdistdof = (k-1, (k-1)*(N-1))

    return (f, k-1), (im, fdistdof)

def compute_CD(avranks, N, alpha="0.05", type="nemenyi"):
    """
    if type == "nemenyi":
        critical difference for Nemenyi two tailed test.
    if type == "bonferroni-dunn":
        critical difference for Bonferroni-Dunn test
    """

    k = len(avranks)
   
    d = {}

    d[("nemenyi", "0.05")] = [0, 0, 1.960, 2.343, 2.568, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164 ]
    d[("nemenyi", "0.1")] = [0, 0, 1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920 ]    
    d[("bonferroni-dunn", "0.05")] =  [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576, 2.638, 2.690, 2.724, 2.773 ]
    d[("bonferroni-dunn", "0.1")] = [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326, 2.394, 2.450, 2.498, 2.539 ]

    q = d[(type, alpha)]

    cd = q[k]*(k*(k+1)/(6.0*N))**0.5

    return cd
 

def graph_ranks(filename, avranks, names, cd=None, cdmethod=None, lowv=None, highv=None, width=6, textspace=1, reverse=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods' 
    performance.
    See Janez Demsar, Statistical Comparisons of Classifiers over 
    Multiple Data Sets, 7(Jan):1--30, 2006. 

    Needs matplotlib to work.

    Arguments:
    filename -- Output file name (with extension). Formats supported
        by matplotlib can be used.
    avranks -- List of average methods' ranks.
    names -- List of methods' names.

    Keyword arguments:
    cd -- Critical difference. Used for marking methods that whose
        difference is not statistically significant.
    lowv -- The lowest shown rank, if None, use 1.
    highv -- The highest shown rank, if None, use len(avranks).
    width -- Width of the drawn figure in inches, default 6 in.
    textspace -- Space on figure sides left for the description
        of methods, default 1 in.
    reverse -- If True, the lowest rank is on the right. Default:
        False.
    cdmethod -- None by default. It can be an index of element in avranks or
        or names which specifies the method which should be marked
        with an interval.

    Maintainer: Marko Toplak
    """

    width = float(width)
    textspace = float(textspace)

    def nth(l,n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l,n)
        return [ a[n] for a in l ]

    def lloc(l,n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0])+n
        else:
            return n 

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5]) 
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if len(lr) == 0:
            yield ()
        else:
            #it can work with single numbers
            index = lr[0]
            if type(1) == type(index):
                index = [ index ]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    try:
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except:
        import sys
        print >> sys.stderr, "Function requires matplotlib. Please install it."
        return

    def printFigure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort =  sorted([ (a,i) for i,a in  enumerate(sums) ], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [ names[x] for x in sortidx ]
    
    if not lowv:
        lowv = min(1, int(math.floor(min(ssums))))
    if not highv:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None
    sums = sorted(sums)

    linesblank = 0
    scalewidth = width - 2*textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        
        return textspace+scalewidth/(highv-lowv)*a

    distanceh = 0.25

    if cd and cdmethod == None:
    
        #get pairs of non significant methods

        def getLines(sums, hsd):

            #get all pairs
            lsums = len(sums)
            allpairs = [ (i,j) for i,j in mxrange([[lsums], [lsums]]) if j > i ]
            #remove not significant
            notSig = [ (i,j) for i,j in allpairs if abs(sums[i]-sums[j]) <= hsd ]
            #keep only longest
            
            def noLonger((i,j), notSig):
                for i1,j1 in notSig:
                    if (i1 <= i and j1 > j) or (i1 < i and j1 >= j):
                        return False
                return True

            longest = [ (i,j) for i,j in notSig if noLonger((i,j),notSig) ]
            
            return longest

        lines = getLines(ssums, cd)
        linesblank = 0.2 + 0.2 + (len(lines)-1)*0.1

        #add scale
        distanceh = 0.25
        cline += distanceh

    #calculate height needed height of an image
    minnotsignificant = max(2*0.2, linesblank)
    height = cline + ((k+1)/2)*0.2 + minnotsignificant

    fig = Figure(figsize=(width, height))
    ax = fig.add_axes([0,0,1,1]) #reverse y axis
    ax.set_axis_off()

    hf = 1./height # height factor
    wf = 1./width

    def hfl(l): 
        return [ a*hf for a in l ]

    def wfl(l): 
        return [ a*wf for a in l ]

    """
    Upper left corner is (0,0).
    """

    ax.plot([0,1], [0,1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l,0)), hfl(nth(l,1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf*x, hf*y, s, *args, **kwargs)

    line([(textspace, cline), (width-textspace, cline)], linewidth=0.7)
    
    bigtick = 0.1
    smalltick = 0.05


    import numpy

    for a in list(numpy.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a): tick = bigtick
        line([(rankpos(a), cline-tick/2),(rankpos(a), cline)], linewidth=0.7)

    for a in range(lowv, highv+1):
        text(rankpos(a), cline-tick/2-0.05, str(a), ha="center", va="bottom")

    k = len(ssums)

    for i in range((k+1)/2):
        chei = cline+ minnotsignificant + (i)*0.2
        line([(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textspace-0.1, chei)], linewidth=0.7)
        text(textspace-0.2, chei, nnames[i], ha="right", va="center")

    for i in range((k+1)/2, k):
        chei = cline + minnotsignificant + (k-i-1)*0.2
        line([(rankpos(ssums[i]), cline), (rankpos(ssums[i]), chei), (textspace+scalewidth+0.1, chei)], linewidth=0.7)
        text(textspace+scalewidth+0.2, chei, nnames[i], ha="left", va="center")

    if cd and cdmethod == None:

        #upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv+cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)
            
        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick/2), (begin, distanceh - bigtick/2)], linewidth=0.7)
        line([(end, distanceh + bigtick/2), (end, distanceh - bigtick/2)], linewidth=0.7)
        text((begin+end)/2, distanceh - 0.05, "CD", ha="center", va="bottom")

        #non significance lines    
        def drawLines(lines, side=0.05, height=0.1):
            start = cline + 0.2
            for l,r in lines:  
                line([(rankpos(ssums[l])-side, start), (rankpos(ssums[r])+side, start)], linewidth=2.5) 
                start += height

        drawLines(lines)

    elif cd:
        begin = rankpos(avranks[cdmethod]-cd)
        end = rankpos(avranks[cdmethod]+cd)
        line([(begin, cline), (end, cline)], linewidth=2.5) 
        line([(begin, cline + bigtick/2), (begin, cline - bigtick/2)], linewidth=2.5)
        line([(end, cline + bigtick/2), (end, cline - bigtick/2)], linewidth=2.5)
 
    printFigure(fig, filename, **kwargs)

if __name__ == "__main__":
    avranks =  [3.143, 2.000, 2.893, 1.964]
    names = ["prva", "druga", "tretja", "cetrta" ]
    cd = compute_CD(avranks, 14)
    #cd = compute_CD(avranks, 10, type="bonferroni-dunn")
    print cd

    print compute_friedman(avranks, 14)

    #graph_ranks("test.eps", avranks, names, cd=cd, cdmethod=0, width=6, textspace=1.5)
