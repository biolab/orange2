import statc, operator, math
import orngMisc, orngTest


#### Private stuff

def log2(x):
    return math.log(x)/math.log(2)

def checkNonZero(x):
    if x==0.0:
        raise SystemError, "cannot compute: no examples or sum of weight is 0.0"

def gettotweight(res):
    totweight = reduce(lambda x, y: x+y.weight, res.results, 0)
    if totweight==0.0:
        raise SystemError, "cannot compute: sum of weights is 0.0"
    return totweight

def gettotsize(res):
    if len(res.results):
        return len(res.results)
    else:
        raise SystemError, "cannot compute: no examples"

def gettotconfm(confm):
    tot = confm.TP + confm.FP + confm.TN + confm.FN
    checkNonZero(tot)
    return tot

def sqr(x):
    return x*x


#### Utility

def splitByIterations(res):
    ress = [orngTest.ExperimentResults(1, res.numberOfLearners, res.weights, classifiers=res.classifiers, loaded=res.loaded)
            for i in range(res.numberOfIterations)]
    for te in res.results:
        ress[te.iterationNumber].results.append(te)
    return ress    

    
#### Statistics

def CA(res, **argkw):
    if type(res)==ConfusionMatrix:
        div = nm.TP+nm.FN+nm.FP+nm.TN
        checkNonZero(div)
        return (nm.TP+nm.TN)/div
        
    else:
        CAs = [0.0]*res.numberOfLearners

        if argkw.get("unweighted", 0) or not res.weights:
            for tex in res.results:
                CAs = map(lambda res, cls: res+(cls==tex.actualClass), CAs, tex.classes)
            totweight = gettotsize(res)
        else:
            for tex in res.results:
                CAs = map(lambda res, cls: res+(cls==tex.actualClass and tex.weight), CAs, tex.classes)
            totweight = gettotweight(res)

        return [x/totweight for x in CAs]

    
def MSE(res, **argkw):
    MSEs = [0.0]*res.numberOfLearners

    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            MSEs = map(lambda res, cls, ac = tex.actualClass:
                       res + sqr(cls - ac), MSEs, tex.classes)
        totweight = gettotsize(res)
    else:
        for tex in res.results:
            MSEs = map(lambda res, cls, ac = tex.actualClass, tw = tex.weight:
                       res + tw*sqr(cls - ac), MSEs, tex.classes)
        totweight = gettotweight(res)

    return [x/totweight for x in MSEs]

    
def CA_se(res, **argkw):
    if res.numberOfIterations==1:
        if argkw.get("unweighted", 0) or not res.weights:
            totweight = gettotsize(res)
        else:
            totweight = gettotweight(res)
        return [(x, x*(1-x)/math.sqrt(totweight)) for x in apply(CA, (res,), argkw)]
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

        newFolds = []
        for lrn in range(res.numberOfLearners):
            newF = []
            for fold in range(res.numberOfIterations):
                if foldN[fold]>0.0:
                        newF.append(CAsByFold[lrn][fold]/foldN[fold])
            newFolds.append(newF)

        checkNonZero(len(newFolds))
        return [(statc.mean(cas), statc.sterr(cas)) for cas in newFolds]



def AP(res, **argkw):
    APs=[0.0]*res.numberOfLearners
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            APs = map(lambda res, probs: res + probs[tex.actualClass], APs, tex.probabilities)
        totweight = gettotsize(res)
    else:
        for tex in res.results:
            APs = map(lambda res, probs: res + probs[tex.actualClass]*tex.weight, APs, tex.probabilities)
        totweight = gettotweight(res)
    return [AP/totweight for AP in APs]


def BrierScore(res, **argkw):
    # Computes an average (over examples) of sum_x(t(x) - p(x))^2, where
    #    x is class,
    #    t(x) is 0 for 'wrong' and 1 for 'correct' class
    #    p(x) is predicted probabilty.
    # There's a trick: since t(x) is zero for all classes but the
    # correct one (c), we compute the sum as sum_x(p(x)^2) - 2*p(c) + 1
    # Since +1 is there for each example, it add 1 to the average
    # We skip the +1 inside the sum and add it just at the end of the function
    MSEs=[0.0]*res.numberOfLearners
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            MSEs = map(lambda res, probs:
                       res + reduce(lambda s, pi: s+sqr(pi), probs, 0) - 2*probs[tex.actualClass], MSEs, tex.probabilities)
        totweight = gettotsize(res)
    else:
        for tex in res.results:
            MSEs = map(lambda res, probs:
                       res + tex.weight*reduce(lambda s, pi: s+sqr(pi), probs, 0) - 2*probs[tex.actualClass], MSEs, tex.probabilities)
        totweight = gettotweight(res)
    return [x/totweight+1.0 for x in MSEs]


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
    

def aprioriDistributions(res, **argkw):
    probs = [0.0]*(max([x.actualClass for x in res.results])+1)
    if argkw.get("unweighted", 0) or not res.weights:
        for tex in res.results:
            probs[tex.actualClass] += 1.0
        totweight = gettotsize(res)
    else:
        for tex in res.results:
            probs[tex.actualClass] += tex.weight
        totweight = gettotweight(res)
    return [prob/totweight for prob in probs]

    
def IS(res, apriori=None, **argkw):
    if not apriori:
        apriori = aprioriDistributions(res)
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
    return [IS/totweight for IS in ISs]




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


def computeConfusionMatrices(res, classIndex=-1, **argkw):
    tfpns = [ConfusionMatrix() for i in range(res.numberOfLearners)]

    if classIndex<0:
        if res.baseClass>=0:
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


def sens(confm):
    if type(confm) == list:
        return [sens(cm) for cm in confm]
    else:
        tot = confm.TP+confm.FN
        checkNonZero(tot)
        return confm.TP/tot


def spec(confm):
    if type(confm) == list:
        return [spec(cm) for cm in confm]
    else:
        tot = confm.FP+confm.TN
        checkNonZero(tot)
        return confm.TN/tot
  

def PPV(confm):
    if type(confm) == list:
        return [PPV(cm) for cm in confm]
    else:
        tot = confm.TP+confm.TN
        checkNonZero(tot)
        return confm.TP/tot


def NPV(confm):
    if type(confm) == list:
        return [NPV(cm) for cm in confm]
    else:
        tot = confm.TP+confm.TN
        checkNonZero(tot)
        return confm.TP/tot



def AROC(res, classIndex=-1):
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
            Q2 += thisPos * (sqr(lowNeg)  + lowNeg*thisNeg  + sqr(thisNeg)/3.)
            Q1 += thisNeg * (sqr(highPos) + highPos*thisPos + sqr(thisPos)/3.)


            lowNeg += thisNeg

        W  /= (totPos*totNeg)
        Q1 /= (totNeg*sqr(totPos))
        Q2 /= (totPos*sqr(totNeg))

        SE = math.sqrt( (W*(1-W) + (totPos-1)*(Q1-sqr(W)) + (totNeg-1)*(Q2-sqr(W))) / (totPos*totNeg) )
        results.append((W, SE))
    return results

    
def compare2AROCs(res, lrn1, lrn2, classIndex=-1, **argkw):
    import corn
    return corn.compare2ROCs(res, lrn1, lrn2, classIndex, res.weights and not argkw.get("unweighted"))

    
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


class CDT:
  """ Stores number of concordant (C), discordant (D) and tied (T) pairs (used for aROC) """
  def __init__(self, C=0.0, D=0.0, T=0.0):
    self.C, self.D, self.T = C, D, T
   

def computeCDT(res, classIndex=-1, **argkw):
    if classIndex<0:
        if res.baseClass>=0:
            classIndex = res.baseClass
        else:
            classIndex = 1
            
    import corn
    useweights = res.weights and not argkw.get("unweighted", 0)

    if (res.numberOfIterations>1):
        CDTs = [CDT() for i in range(res.numberOfLearners)]
        iterationExperiments = splitByIterations(res)
        for exp in iterationExperiments:
            expCDTs = corn.computeCDT(exp, classIndex, useweights)
            for i in range(len(CDTs)):
                CDTs[i].C += expCDTs[i].C
                CDTs[i].D += expCDTs[i].D
                CDTs[i].T += expCDTs[i].T
        return CDTs
    else:
        return corn.computeCDT(res, classIndex, useweights)
    
   
def AROCFromCDT(cdt, **argkw):
    if type(cdt) == list:
        return [AROCFromCDT(c) for c in cdt]

    C, D, T = cdt.C, cdt.D, cdt.T
    N = C+D+T
    checkNonZero(N)
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

def AUCFromCDT(cdt, **argkw):
    aucs = apply(AROCFromCDT, (cdt,), argkw)
    if type(cdt) == list:
        return [x[-1] for x in aucs]
    else:
        return aucs[-1]


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
                mcm[l2][l1] = sqr(abs(mcm[l1][l2]-mcm[l2][l1])-1) / su
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
        return sqr(abs(tf-ft)-1) / su
    else:
        return 0



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
    apply(learningCurve2PiCTeX, (file, allResults, proportions), options)
    
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
                    #print wanted, best
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
  apply(legend2PiCTeX, (file, [orngMisc.getobjectname(learners[i], "Learner %i" % i) for i in range(len(learners))]), options)
    
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
