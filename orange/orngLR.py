import orange
import orngCI
import math, os
from Numeric import *
from LinearAlgebra import *


#######################
## Print out methods ##
#######################

def printOUT(classifier):
    # print out class values
    print
    print "class attribute = " + classifier.domain.classVar.name
    print "class values = " + str(classifier.domain.classVar.values)
    print
    
    # get the longest attribute name
    longest=0
    for at in classifier.domain.attributes:
        if len(at.name)>longest:
            longest=len(at.name);

    # print out the head
    formatstr = "%"+str(longest)+"s %18s %10s %10s %10s %10s"
    print formatstr % ("Attribute", "beta", "st. error", "wald Z", "P", "OR=exp(beta)")
    print
    formatstr = "%"+str(longest)+"s %10.10f %10.2f %10.2f %10.2f"    
    print formatstr % ("Intercept", classifier.beta[0], classifier.beta_se[0], classifier.wald_Z[0], classifier.P[0])
    formatstr = "%"+str(longest)+"s %10.10f %10.2f %10.2f %10.2f %10.2f"    
    for i in range(len(classifier.domain.attributes)):
        print formatstr % (classifier.domain.attributes[i].name, classifier.beta[i+1], classifier.beta_se[i+1], classifier.wald_Z[i+1], abs(classifier.P[i+1]), exp(classifier.beta[i+1]))
        


##########################
## LEARNER improvements ##
##########################
#construct "continuous" attributes from discrete attributes
def createNoDiscDomain(domain):
    attributes = []
    #iterate through domain
    for at in domain.attributes:
        #if att is discrete, create (numOfValues)-1 new ones and set getValueFrom
        if at.varType == orange.VarTypes.Discrete:
            for ival in range(len(at.values)):
                # continue at first value 
                if ival == 0:
                    continue
                # create attribute
                newVar = orange.FloatVariable(at.name+"="+at.values[ival])
                
                # create classifier
                vals = [orange.Value((float)(ival==i)) for i in range(len(at.values))]
                vals.append("?")
                #print (vals)
                cl = orange.ClassifierByLookupTable(newVar, at, vals)                
                newVar.getValueFrom=cl

                # append newVariable                
                attributes.append(newVar)
        else:
            # add original attribute
            attributes.append(at)
    attributes.append(domain.classVar)
    retDomain = orange.Domain(attributes)
    for k in domain.getmetas().keys():
        retDomain.addmeta(orange.newmetaid(), domain.getmetas()[k])
    return retDomain

def createFullNoDiscDomain(domain):
    attributes = []
    #iterate through domain
    for at in domain.attributes:
        #if att is discrete, create (numOfValues)-1 new ones and set getValueFrom
        if at.varType == orange.VarTypes.Discrete:
            for ival in range(len(at.values)):
                # create attribute
                newVar = orange.FloatVariable(at.name+"="+at.values[ival])
                
                # create classifier
                vals = [orange.Value((float)(ival==i)) for i in range(len(at.values))]
                vals.append("?")
                #print (vals)
                cl = orange.ClassifierByLookupTable(newVar, at, vals)                
                newVar.getValueFrom=cl

                # append newVariable                
                attributes.append(newVar)
        else:
            # add original attribute
            attributes.append(at)
    attributes.append(domain.classVar)
    return orange.Domain(attributes)
                
# returns data set without discrete values. 
def createNoDiscTable(olddata):
    newdomain = createNoDiscDomain(olddata.domain)
    #print newdomain
    return olddata.select(newdomain)

def createFullNoDiscTable(olddata):
    newdomain = createFullNoDiscDomain(olddata.domain)
    #print newdomain
    return olddata.select(newdomain)
    

def hasDiscreteValues(domain):
    for at in domain.attributes:
        if at.varType == orange.VarTypes.Discrete:
            return 1
    return 0

def LogRegLearner(examples = None, weightID=0, **kwds):
    lr = LogRegLearnerClass(**kwds)
    if examples:
        return lr(examples, weightID)
    else:
        return lr

class LogRegLearnerClass:
    def __init__(self, removeSingular=0, **kwds):
        self.__dict__ = kwds
        print removeSingular
        self.removeSingular = removeSingular
    def __call__(self, examples, weight=0):
        nexamples = orange.Preprocessor_dropMissing(examples)
        if hasDiscreteValues(examples.domain):
            nexamples = createNoDiscTable(nexamples)
        else:
            nexamples = nexamples

        learner = orange.LogRegLearner()

        #if self.fitter:
            #learner.fitter = self.fitter
            
        if self.removeSingular:
            lr = learner.fitModel(nexamples, weight)
        else:
            lr = learner(nexamples, weight)
        while isinstance(lr,orange.Variable):
            nexamples.domain.attributes.remove(lr)
            nexamples = nexamples.select(orange.Domain(nexamples.domain.attributes, nexamples.domain.classVar))
            lr = learner.fitModel(nexamples, weight)
        return lr


def Univariate_LogRegLearner(examples=None, **kwds):
    learner = apply(Univariate_LogRegLearner_Class, (), kwds)
    if examples:
        return learner(examples)
    else:
        return learner

class Univariate_LogRegLearner_Class:
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, examples):
        examples = createFullNoDiscTable(examples)
        classifiers = map(lambda x: LogRegLearner(orange.Preprocessor_dropMissing(examples.select(orange.Domain(x, examples.domain.classVar)))), examples.domain.attributes)
        maj_classifier = LogRegLearner(orange.Preprocessor_dropMissing(examples.select(orange.Domain(examples.domain.classVar))))
        beta = [maj_classifier.beta[0]] + [x.beta[1] for x in classifiers]
        beta_se = [maj_classifier.beta_se[0]] + [x.beta_se[1] for x in classifiers]
        P = [maj_classifier.P[0]] + [x.P[1] for x in classifiers]
        wald_Z = [maj_classifier.wald_Z[0]] + [x.wald_Z[1] for x in classifiers]
        domain = examples.domain

        return Univariate_LogRegClassifier(beta = beta, beta_se = beta_se, P = P, wald_Z = wald_Z, domain = domain)

class Univariate_LogRegClassifier:
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, example, resultType = orange.GetValue):
        # classification not implemented yet. For now its use is only to provide regression coefficients and its statistics
        pass
    

def LogRegLearner_getPriors(examples = None, weightID=0, **kwds):
    lr = LogRegLearnerClass_getPriors(**kwds)
    if examples:
        return lr(examples, weightID)
    else:
        return lr

class LogRegLearnerClass_getPriors:
    def __init__(self, removeSingular=0, **kwds):
        self.__dict__ = kwds
        self.removeSingular = removeSingular
    def __call__(self, examples, weight=0):
        # next function changes data set to a extended with unknown values 
        def createLogRegExampleTable(data):
            newDomain = orange.Domain(data.domain.attributes+[data.domain.classVar])
            newDomain.addmeta(orange.newmetaid(), orange.FloatVariable("weight"))
            dataOrig = data.select(newDomain) #original data
            dataFinal = dataOrig.select(newDomain) #final results will be stored in this object

            for d in dataFinal:
                d["weight"]=1000000.

            for at in data.domain.attributes:
                # za vsak atribut kreiraj nov newExampleTable newData
                newData = orange.ExampleTable(dataOrig)
                
                # v dataOrig, dataFinal in newData dodaj nov atribut -- continuous variable
                if at.varType == orange.VarTypes.Continuous:
                    atDisc = orange.FloatVariable(at.name + "Disc")
                    newDomain = orange.Domain(dataOrig.domain.attributes+[atDisc,dataOrig.domain.classVar])
                    for (id, metaVar) in dataOrig.domain.getmetas().items():
                        newDomain.addmeta(id, metaVar)
                    dataOrig = dataOrig.select(newDomain)
                    dataFinal = dataFinal.select(newDomain)
                    newData = newData.select(newDomain)
                    for d in dataOrig:
                        d[atDisc] = 0
                    for d in dataFinal:
                        d[atDisc] = 0
                    for d in newData:
                        d[atDisc] = 1
                        d[at] = 0
                
                # v dataOrig, dataFinal in newData atributu "at" dodaj ee  eno  vreednost, ki ima vrednost kar  ime atributa +  "X"
                if at.varType == orange.VarTypes.Discrete:
                    dataOrigOld = orange.ExampleTable(dataOrig)
                    dataFinalOld = orange.ExampleTable(dataFinal)
                    atNew = orange.EnumVariable(at.name, values = at.values + [at.name+"X"])
                    newDomain = orange.Domain(filter(lambda x: x!=at, dataOrig.domain.attributes)+[atNew,dataOrig.domain.classVar])
                    for (id, metaVar) in dataOrig.domain.getmetas().items():
                        newDomain.addmeta(id, metaVar)
                    dataOrig = dataOrig.select(newDomain)
                    dataFinal = dataFinal.select(newDomain)
                    newData = newData.select(newDomain)
                    for d in range(len(dataOrig)):
                        dataOrig[d][atNew] = dataOrigOld[d][at]
                    for d in range(len(dataFinal)):
                        dataFinal[d][atNew] = dataFinalOld[d][at]
                    for d in newData:
                        d[atNew] = at.name+"X"

                for at in newData:
                    at["weight"]=0.1

                # v newData doloci temu atributu vrednost 1, v drugih dveh pa 0---DONEW
                # skopiraj vse vrednosti iz newData v dataFinal
                for d in newData:
                    dataFinal.append(orange.Example(d))
                    
            return dataFinal            
        def findZero(model, at):
            if at.varType == orange.VarTypes.Discrete:
                for i_a in range(len(model.domain.attributes)):
                    if model.domain.attributes[i_a].name == at.name+"="+at.name+"X":
                        return model.beta[i_a+1]
            else:
                for i_a in range(len(model.domain.attributes)):
                    if model.domain.attributes[i_a].name == at.name+"Disc":
                        return model.beta[i_a+1]
                    
        nexamples = orange.Preprocessor_dropMissing(examples)
        # get Original Model
        orig_model = LogRegLearner(examples)

        # get extended Model (you should not change data)
        extended_examples = createLogRegExampleTable(examples)
        extended_model = LogRegLearner(extended_examples, extended_examples.domain.getmeta("weight"))

        print "domains", orig_model.domain, extended_model.domain
        # izracunas odstopanja
        # get sum of all betas
        beta = 0
        betas_ap = []
        for at in range(len(nexamples.domain.attributes)):
            att = nexamples.domain.attributes[at]    
            beta_add = findZero(extended_model, att)
            betas_ap.append(beta_add)
            beta = beta + beta_add
        
        # substract it from intercept
        logistic_prior = extended_model.beta[0]+beta
        
        # compare it to bayes prior
        bayes = orange.BayesLearner(nexamples)
        bayes_prior = math.log(bayes.distribution[1]/bayes.distribution[0])

        # normalize errors
        k = (bayes_prior-extended_model.beta[0])/(logistic_prior-extended_model.beta[0])

        betas_ap = [k*x for x in betas_ap]                

        # vrni originalni model in pripadajoce apriorne niclele
        print "returnam ZDEJ!", orig_model, betas_ap
        return (orig_model, betas_ap)


######################################
#### Fitters for logistic regression (logreg) learner ####
######################################

def Pr(x, betas):
    k = math.exp(dot(x, betas))
    return k / (1+k)

def lh(x,y,betas):
    return 0

class simpleFitter(orange.LogRegFitter):
    def __init__(self, penalty=0):
        self.penalty = penalty
    def __call__(self, data, weight=0):
        ml = data.native(0)
        for i in range(len(data.domain.attributes)):
          a = data.domain.attributes[i]
          if a.varType == orange.VarTypes.Discrete:
            for m in ml:
              m[i] = a.values.index(m[i])
        for m in ml:
          m[-1] = data.domain.classVar.values.index(m[-1])

        Xtmp = array(ml)
        y = Xtmp[:,-1]   # true probabilities (1's or 0's)
        one = reshape(array([1]*len(data)), (len(data),1)) # intercept column
        X=concatenate((one, Xtmp[:,:-1]),1)  # intercept first, then data

        betas = array([0.0] * (len(data.domain.attributes)+1))

# predict the probability for an instance, x and betas are vectors


# start the computation

        N = len(data)
        for i in range(20):
            p = array([Pr(X[i], betas) for i in range(len(data))])

            W = identity(len(data), Float)
            pp = p * (1.0-p)
            for i in range(N):
                W[i,i] = pp[i]

            WI = inverse(W)
            z = matrixmultiply(X, betas) + matrixmultiply(WI, y - p)

            tmpA = inverse(matrixmultiply(transpose(X), matrixmultiply(W, X))+self.penalty*identity(len(data.domain.attributes)+1, Float))
            tmpB = matrixmultiply(transpose(X), matrixmultiply(W, z))
            betas = matrixmultiply(tmpA, tmpB)
            likelihood_new = lh(X,y,betas)
            #if abs(likelihood_new-likelihood)<0.001:
            #    break
            likelihood = likelihood_new
            
        XX = sqrt(diagonal(inverse(matrixmultiply(transpose(X),X))))
        yhat = array([Pr(X[i], betas) for i in range(len(data))])
        ss = sum((y - yhat) ** 2) / (N - len(data.domain.attributes) - 1)
        sigma = math.sqrt(ss)
        beta = []
        beta_se = []
        for i in range(len(betas)):
            beta.append(betas[i])
            beta_se.append(0.0)
        return (self.OK, beta, beta_se, 0)




    
############################################################
####  Feature subset selection for logistic regression  ####
############################################################


def StepWiseFSS(examples = None, **kwds):
    """
      Constructs and returns a new set of examples that includes a
      class and attributes selected by stepwise logistic regression. This is an
      implementation of algorithm described in [Hosmer and Lemeshow, Applied Logistic Regression, 2000]

      examples: data set (ExampleTable)     
      addCrit: "Alpha" level to judge if variable has enough importance to be added in the new set. (e.g. if addCrit is 0.2, then attribute is added if its P is lower than 0.2)
      deleteCrit: Similar to addCrit, just that it is used at backward elimination. It should be higher than addCrit!
      numAttr: maximum number of selected attributes, use -1 for infinity
    """

    fss = apply(StepWiseFSS_class, (), kwds)
    if examples:
        return fss(examples)
    else:
        return fss

def getLikelihood(fitter, examples):
    res = fitter(examples)
    if res[0] in [fitter.OK, fitter.Infinity, fitter.Divergence]:
       status, beta, beta_se, likelihood = res
       return likelihood
    else:
       return -100*len(examples)
        
    

class StepWiseFSS_class:
  def __init__(self, addCrit=0.2, deleteCrit=0.3, numAttr = -1):
    self.addCrit = addCrit
    self.deleteCrit = deleteCrit
    self.numAttr = numAttr
  def __call__(self, examples):
    attr = []
    remain_attr = examples.domain.attributes[:]

    print self.addCrit
    print self.deleteCrit
    
    # get LL for Majority Learner 
    tempDomain = orange.Domain(attr,examples.domain.classVar)
    tempData  = createNoDiscTable(orange.Preprocessor_dropMissing(examples.select(tempDomain)))

    ll_Old = getLikelihood(orange.LogRegFitter_Cholesky(), tempData)
    length_Old = len(tempData)

    stop = 0
    while not stop:
        # LOOP until all variables are added or no further deletion nor addition of attribute is possible
        
        # if there are more than 1 attribute then perform backward elimination
        if len(attr) >= 2:
            minG = 1000
            worstAt = attr[0]
            ll_Best = ll_Old
            length_Best = length_Old
            for at in attr:
                # check all attribute whether its presence enough increases LL?

                tempAttr = filter(lambda x: x!=at, attr)
                tempDomain = orange.Domain(tempAttr,examples.domain.classVar)
                # domain, calculate P for LL improvement.
                tempData  = createNoDiscTable(orange.Preprocessor_dropMissing(examples.select(tempDomain)))
                ll_Delete = getLikelihood(orange.LogRegFitter_Cholesky(), tempData)
                length_Delete = len(tempData)
                # P=PR(CHI^2>G), G=-2(L(0)-L(1))=2(E(0)-E(1))
                length_Avg = (length_Delete + length_Old)/2.0


                G=-2*length_Avg*(ll_Delete/length_Delete-ll_Old/length_Old)

                print tempDomain
                print G
                print length_Avg*ll_Delete/length_Delete
                print length_Avg*ll_Old/length_Old
                # set new best attribute                
                if G<minG:
                    worstAt = at
                    minG=G
                    ll_Best = ll_Delete
                    length_Best = length_Delete
            # deletion of attribute

            if worstAt.varType==orange.VarTypes.Continuous:
                P=lchisqprob(minG,1);
            else:
                P=lchisqprob(minG,len(worstAt.values)-1);
            print P
            if P>=self.deleteCrit:
                attr.remove(worstAt)
                remain_attr.append(worstAt)
                nodeletion=0
                ll_Old = ll_Best
                length_Old = length_Best
            else:
                nodeletion=1
        else:
            nodeletion = 1
            # END OF DELETION PART
            
        # if enough attributes has been chosen, stop the procedure
        if self.numAttr>-1 and len(attr)>=self.numAttr:
            remain_attr=[]
         
        # for each attribute in the remaining
        maxG=-1
        ll_Best = ll_Old
        length_Best = length_Old
        for at in remain_attr:
            tempAttr = attr + [at]
            tempDomain = orange.Domain(tempAttr,examples.domain.classVar)
            # domain, calculate P for LL improvement.
            tempData  = createNoDiscTable(orange.Preprocessor_dropMissing(examples.select(tempDomain)))
            ll_New = getLikelihood(orange.LogRegFitter_Cholesky(), tempData)

            length_New = len(tempData) # get number of examples in tempData to normalize likelihood

            # P=PR(CHI^2>G), G=-2(L(0)-L(1))=2(E(0)-E(1))

            length_avg = (length_New + length_Old)/2
            G=-2*length_avg*(ll_Old/length_Old-ll_New/length_New);
            if G>maxG:
                bestAt = at
                maxG=G
                ll_Best = ll_New
                length_Best = length_New
                
        if bestAt.varType==orange.VarTypes.Continuous:
            P=lchisqprob(maxG,1);
        else:
            P=lchisqprob(maxG,len(bestAt.values)-1);
        # Add attribute with smallest P to attributes(attr)
        if P<=self.addCrit:
            attr.append(bestAt)
            remain_attr.remove(bestAt)
            ll_Old = ll_Best
            length_Old = length_Best

        if P>self.addCrit and nodeletion:
            stop = 1

    #print "Likelihood is:"
    #print ll_Old
    #return examples.select(orange.Domain(attr,examples.domain.classVar))
    return attr


def StepWiseFSS_Filter(examples = None, **kwds):
    """
        check function StepWiseFSS()
    """

    filter = apply(StepWiseFSS_Filter_class, (), kwds)
    if examples:
        return filter(examples)
    else:
        return filter


class StepWiseFSS_Filter_class:
    def __init__(self, addCrit=0.2, deleteCrit=0.3, numAttr = -1):
        self.addCrit = addCrit
        self.deleteCrit = deleteCrit
        self.numAttr = numAttr
    def __call__(self, examples):
        attr = StepWiseFSS(examples, addCrit=self.addCrit, deleteCrit = self.deleteCrit, numAttr = self.numAttr)
        return examples.select(orange.Domain(attr, examples.domain.classVar))
                

####################################
####  PROBABILITY CALCULATIONS  ####
####################################

def lchisqprob(chisq,df):
    """
Returns the (1-tailed) probability value associated with the provided
chi-square value and df.  Adapted from chisq.c in Gary Perlman's |Stat.

Usage:   lchisqprob(chisq,df)
"""
    BIG = 20.0
    def ex(x):
    	BIG = 20.0
    	if x < -BIG:
    	    return 0.0
    	else:
    	    return math.exp(x)
    if chisq <=0 or df < 1:
    	return 1.0
    a = 0.5 * chisq
    if df%2 == 0:
    	even = 1
    else:
    	even = 0
    if df > 1:
    	y = ex(-a)
    if even:
    	s = y
    else:
        s = 2.0 * zprob(-math.sqrt(chisq))
    if (df > 2):
        chisq = 0.5 * (df - 1.0)
        if even:
            z = 1.0
        else:
            z = 0.5
        if a > BIG:
            if even:
            	e = 0.0
            else:
            	e = math.log(math.sqrt(math.pi))
            c = math.log(a)
            while (z <= chisq):
            	e = math.log(z) + e
            	s = s + ex(c*z-a-e)
            	z = z + 1.0
            return s
        else:
            if even:
                e = 1.0
            else:
                e = 1.0 / math.sqrt(math.pi) / math.sqrt(a)
    		c = 0.0
    		while (z <= chisq):
    		    e = e * (a/float(z))
    		    c = c + e
    		    z = z + 1.0
    		return (c*y+s)
    else:
    	return s


def zprob(z):
    """
Returns the area under the normal curve 'to the left of' the given z value.
Thus, 
    for z<0, zprob(z) = 1-tail probability
    for z>0, 1.0-zprob(z) = 1-tail probability
    for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability
Adapted from z.c in Gary Perlman's |Stat.

Usage:   lzprob(z)
"""
    Z_MAX = 6.0    # maximum meaningful z-value
    if z == 0.0:
	x = 0.0
    else:
	y = 0.5 * math.fabs(z)
	if y >= (Z_MAX*0.5):
	    x = 1.0
	elif (y < 1.0):
	    w = y*y
	    x = ((((((((0.000124818987 * w
			-0.001075204047) * w +0.005198775019) * w
		      -0.019198292004) * w +0.059054035642) * w
		    -0.151968751364) * w +0.319152932694) * w
		  -0.531923007300) * w +0.797884560593) * y * 2.0
	else:
	    y = y - 2.0
	    x = (((((((((((((-0.000045255659 * y
			     +0.000152529290) * y -0.000019538132) * y
			   -0.000676904986) * y +0.001390604284) * y
			 -0.000794620820) * y -0.002034254874) * y
		       +0.006549791214) * y -0.010557625006) * y
		     +0.011630447319) * y -0.009279453341) * y
		   +0.005353579108) * y -0.002141268741) * y
		 +0.000535310849) * y +0.999936657524
    if z > 0.0:
	prob = ((x+1.0)*0.5)
    else:
	prob = ((1.0-x)*0.5)
    return prob

   