#
# Module Orange Linear Model Visualization
# ----------------------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
#
# Purpose: Visualize all linear models (NB, LogReg, linear SVM, perceptron, etc.).
#
# ChangeLog:
#   - 2003/11/17: project initiated

import orange, orngDimRed
import math, Numeric, LinearAlgebra


def _treshold(x):
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    return 0.5

def _avg(set):
    assert(len(set)>0)
    return float(sum(set))/len(set)

class _parse:
    def bucketize(self, examples, attribute, buckets, getbuckets = 0):
        assert(attribute.varType == 2) # continuous
        l = [float(x[attribute]) for x in examples]
        sort(l)
        m = len(l)/buckets
        if m == 0:
            buckets = 1
        sets = [l[:m*buckets]]
        for x in range(1,buckets+1):
            sets.append(l[m*buckets:m*buckets+1])
        values = [_avg(x) for x in sets]
        if getbuckets:
            return (values,sets)
        else:
            return values

    def quantize(self,values,x):
        pv = values[0]
        pi = 0
        for k in xrange(1,len(values)):
            if cv <= values[k]:
                if values[k]-cv < cv-pv:
                    pi = k-1
                    return k
            pv = values[k]
            pi = k
        return len(values)-1
        
        
    def __init__(self):
        pass

class _parseNB(_parse):
    def _safeRatio(self,a,b):
        if a*10000.0 < b:
            return -10
        elif b*10000.0 < a:
            return 10
        else:
            return math.log(a)-math.log(b)

    def __call__(self,classifier,examples, buckets):
        # todo todo - support for loess
        
        beta = -self._safeRatio(classifier.distribution[1],classifier.distribution[0])
        coeffs = []
        coeff_names = []
        offsets = []
        setss = []
        runi = 0
        for i in range(len(classifier.domain.attributes)):
            tc = ["%s"%classifier.domain.attributes[i].name]
            offsets.append(runi)
            if classifier.domain.attributes[i].varType == 1:
                # discrete attribute
                for j in range(len(classifier.domain.attributes[i].values)):
                    tc.append('%s'%(classifier.domain.attributes[i].values[j]))
                    p1 = classifier.conditionalDistributions[i][j][1]
                    p0 = classifier.conditionalDistributions[i][j][0]
                    coeffs.append(self._safeRatio(p1,p0)+beta)
                    runi += 1
                setss.append(0)
            elif classifier.domain.attributes[i].varType == 2:
                # continuous attribute
                (val,sets,avg,stddev) = self._bucketize(examples,classifier.domain.attributes[i],buckets,1)
                p1 = 0.0
                p0 = 0.0
                # fill in the sum of probabilities, finding the nearest probability
                l = 0
                mm = classifier.conditionalDistributions[i].items()
                for j in xrange(len(val)):
                    tc.append(val[j])
                    pp = -1e200
                    pv = -1e200
                    k = 0
                    while k < len(sets[j]) and l < len(mm):
                        # find the two nearest points to the given example value
                        while l < len(mm) and mm[l][0] <= sets[j][k]:
                            pv = mm[l][0]
                            pp = mm[l][1]
                            l += 1
                        # mark example values that are closer to the previous point
                        while k < len(sets[j]) and sets[j][k]-pv <= mm[l][0]-sets[j][k]:
                            p1 += pp[1]
                            p0 += pp[0]
                            k += 1
                    coeffs.append(self._safeRatio(p1,p0)+beta)
                    runi += 1
                setss.append(sets)
            else:
                raise "unknown attribute type"
            coeff_names.append(tc)
            

        # create the basis vectors for each attribute
        basis = Numeric.identity((len(coeffs)), Numeric.Float)
        for j in range(len(classifier.domain.attributes)):
            if classifier.domain.attributes[j].varType == 2:
                for k in xrange(1,len(coeff_names[j])):
                    i = offsets[j]+k-1
                    basis[i][i] = coeff_names[j][k]
        
        # create the example matrix (only attributes)
        m = Numeric.zeros((len(examples),len(coeffs)), Numeric.Float)
        for i in range(len(examples)):
            for j in range(len(classifier.domain.attributes)):
                if classifier.domain.attributes[j].varType == 1:
                    for k in coeff_names[j][1:]:
                        m[i][offsets[j]+int(examples[i][classifier.domain.attributes[j]])] = 1.0
                else:
                    # quantize
                    cv = float(examples[i][classifier.domain.attributes[j]])
                    k = self._quantize(coeff_names[j][1:],cv)
                    m[i][offsets[j]+k] = cv
                                    
                
        return (beta, coeffs, coeff_names, basis, m, lambda x:math.exp(x)/(1.0+math.exp(x)))

class _parseLR(_parse):
    def getDescriptors(self, trans, examples, buckets):
        tcoeff_names = []
        descriptors = [] # used for managing continuous atts
        for i in range(len(trans)):
            t = trans[i]
            tc = ["%s"%t.attr.name]
            d = t.description()
            #print tc[0],d
            if d[0]==0:
                # continuous                
                values = bucketize(self, examples, t.attr, buckets)
                tc += values
                descriptors.append((i,-1))
            else:
                # nominal
                x = 0
                for n in d[2]:
                    if n!='':
                        tc.append(n)
                        descriptors.append((i,x))
                        x += 1
            tcoeff_names.append(tc)
        return descriptors, tcoeff_names

    def getNames(self, descriptors, tcoeff_names):
        # filter the coeff_names using these masked descriptors
        coeff_names = []
        cur_i = -1
        contins = []
        total = 0
        for (a,b) in descriptors:
            if cur_i == a:
                coeff_names[-1].append(tcoeff_names[a][b+1])
                total += 1
            else:
                if b == -1:
                    # continuous
                    contins.append(len(coeff_names))
                    coeff_names.append(tcoeff_names[a])
                    total += len(coeff_names[-1] - 1)
                else:
                    coeff_names.append([tcoeff_names[a][0],tcoeff_names[a][1+b]])
                    total += 1
                    cur_i = a
        return coeff_names,total,contins

    def getBasis(self, total,xcoeffs,contins):
        # create the basis vectors for each attribute
        basis = Numeric.identity((total), Numeric.Float)
        
        # fix up the continuous attributes in coeffs (duplicate) and in basis (1.0 -> value[i])
        x = Numeric.ones((total,), Numeric.Float)
        coeffs = []
        lookup = []
        offset = 0
        prev = 0
        for i in contins:
            for j in range(prev,i-1):
                # copy from coeffs, do not copy the current value
                lookup.append(len(coeffs))
                coeffs.append(xcoeffs[j])
            prev = i
            vals = coeff_names[i][1:]
            for j in range(len(vals)):
                x[offset+i+j] = vals[j]
            offset += len(vals)-1
        basis *= x
        for j in range(prev,len(xcoeffs)):
            # copy from coeffs, do not copy the current value
            lookup.append(len(coeffs))
            coeffs.append(xcoeffs[j])
        return (basis,lookup,coeffs)

    def getExamples(self,tex,dim,classifier,lookup,contins):
        # create the example matrix (only attributes)
        m = Numeric.zeros((len(tex),dim), Numeric.Float)
        for i in xrange(len(tex)):
            tv = tex[i]
            # do the insertion and quantization
            # copy all coefficients
            for i in xrange(len(lookup)):
                m[i][lookup[i]] = tv[i]

            # quantize the continuous attributes
            for i in contins:
                # obtain the marker
                vals = coeff_names[i][1:]
                x = tv[i]
                newi = self._quantize(vals,x)
                m[i][lookup[i]] = 0.0
                m[i][lookup[i]+newi] = x
        return m

    def __call__(self,classifier,examples, buckets):
        # skip domain translation and masking
        robustc = classifier.classifier
        primitivec = robustc.classifier
        beta = -primitivec.beta[0]

        (descriptors,prevnames) = self.getDescriptors(classifier.translator.trans,examples,buckets)

        # include robust LR's masking
        descriptors = robustc.translate(descriptors)

        (coeff_names, total, contins) = self.getNames(descriptors, prevnames)
        
        xcoeffs = primitivec.beta[1:]

        (basis,lookup,coeffs) = self.getBasis(total, xcoeffs, contins)

        tex = []
        for ex in examples:
            tex.append(robustc.translate(classifier.translator.extransform(ex)))
            
        m = self.getExamples(tex,len(basis),classifier,lookup,contins)

        return (beta, coeffs, coeff_names, basis, m, lambda x:math.exp(x)/(1.0+math.exp(x)))


class _parseSVM(_parseLR):
    def __call__(self,classifier,examples, buckets):
        if classifier.model['kernel_type'] != 0:
            raise "Use SVM with a linear kernel."
        if classifier.model["svm_type"] != 0:
            raise "Use ordinary SVM classification."
        if classifier.model["nr_class"] != 2:
            raise "This is not SVM with a binary class."

        (descriptors,prevnames) = self.getDescriptors(classifier.translate.trans,examples,buckets)

        (coeff_names, total, contins) = self.getNames(descriptors, prevnames)
        
        beta = classifier.model["rho"][0]
        svs = classifier.model["SV"]
        xcoeffs = [0.0]*(len(svs[0])-1)
        for i in xrange(classifier.model["total_sv"]):
            csv = svs[i]
            coef = csv[0][0]
            for j in xrange(len(xcoeffs)):
                assert(csv[j+1][0]-1 == j)
                xcoeffs[j] += coef*csv[j+1][1]

        # reverse the betas if the labels got switched
        if classifier.model["label"][0] == 0:
            beta = -beta
            xcoeffs = [-x for x in xcoeffs]

        (basis,lookup,coeffs) = self.getBasis(total, xcoeffs, contins)

        tex = []
        for i in range(len(examples)):
            tex.append(classifier.translate.extransform(examples[i]))
 
        m = self.getExamples(tex,len(basis),classifier,lookup,contins)

        return (beta, coeffs, coeff_names, basis, m, _treshold)


class _marginConverter:
    def __init__(self,estdomain,estimator):
        self.estdomain = estdomain
        self.cv = self.estdomain.classVar(0)
        self.estimator = estimator

    def __call__(self, r):
        # got a margin
        ex = orange.Example(self.estdomain,[r,self.cv]) # need a dummy class value
        p = self.estimator(ex,orange.GetProbabilities)
#        print "&(",r,p,')'
        return p[1]

class _parseMargin(_parse):
    def __init__(self,marginc,parser):
        self.parser = parser
        self.marginc = marginc

    def __call__(self,classifier,examples, buckets):
        (beta, coeffs, coeff_names, basis, m, _probfunc) = self.parser(classifier.classifier,examples, buckets)
        return (beta, coeffs, coeff_names, basis, m, _marginConverter(self.marginc.estdomain, self.marginc.estimator))
        

class Visualizer:
    def findParser(self, classifier):
        if type(classifier)==orange.BayesClassifier:
             return _parseNB()
        else:
            try:
                name = classifier.name
                if name == 'MarginMetaClassifier':
                    return _parseMargin(classifier,self.findParser(classifier.classifier))
                if name == 'SVM Classifier Wrap':
                    return _parseSVM()
                elif name == 'Basic Logistic Classifier':
                    return _parseLR()
                else:
                    raise ""
            except:
                raise "Unrecognized classifier: %s"%classifier
    
    def __init__(self, examples, classifier, dimensions = 2, buckets = 3):
        # error detection        
        if len(examples.domain.classVar.values) != 2:
            raise "The domain does not have a binary class. Binary class is required."

        all_attributes = [i for i in examples.domain.attributes]+[examples.domain.classVar]
        for i in range(len(all_attributes)):
            for j in range(len(examples)):
                if examples[j][i].isSpecial():
                    raise "A missing value found in instance %d, attribute %s. Missing values are not allowed."%(j,all_attributes[i].name)
                
        # acquire the linear model
        parser = self.findParser(classifier)

        (beta, coeffs, coeff_names, basis, m, probfunc) = parser(classifier,examples, buckets)
        #print "examples:"
        #print m
        #print "basis:"
        #print basis

        # get the parameters of the hyperplane, and normalize it
        n = Numeric.array(coeffs, Numeric.Float)
        length = Numeric.sqrt(Numeric.dot(n,n))
        ilength = 1.0/length
        n *= ilength
        beta *= ilength
        self.probfunc = lambda x:probfunc(x*length)
        #self.probfunc = lambda x:probfunc(x)
        
        #if len(basis) > len(examples):
        #    raise "Too few examples for that many attribute values. This can be handled, ask Aleks to eliminate null spaces."

        # project the example matrix on the separating hyperplane, removing the displacement
        h_dist = Numeric.dot(m,n)
        h_proj = m - Numeric.dot(Numeric.reshape(h_dist,(len(examples),1)),Numeric.reshape(n,(1,len(coeffs))))
        h_dist -= beta # correct distance

        basis_dist = Numeric.dot(basis,n)
        basis_proj = basis - Numeric.dot(Numeric.reshape(basis_dist,(len(coeffs),1)),Numeric.reshape(n,(1,len(coeffs))))
        basis_dist -= beta

        # perform standardization of attributes
        for i in xrange(len(coeffs)):
            # standardize the examples and return parameters
            (h_proj[:,i],transf) = orngDimRed.VarianceScaling(h_proj[:,i])
            # now transform the basis with the same parameters
            (basis_proj[:,i],transf) = orngDimRed.VarianceScaling(basis_proj[:,i],transf)

        # obtain the second dimension using PCA, but we only need a single principal vector
        pca = orngDimRed.PCA(h_proj,1)
        # now transform the basis using the same matrix (need N)
        # U' = U*D*V
        # N' * V^-1 * D^-1 = N
        # N = N' * (D*V)^-1
        DV = Numeric.dot(Numeric.identity(len(coeffs),Numeric.Float)*Numeric.clip(pca.variance,1e-6,1e6),pca.factors)
        nbasis = Numeric.dot(basis_proj,LinearAlgebra.inverse(DV))

        self.coeff_names = coeff_names
        self.coeffs = coeffs
        self.beta = beta

        # coordinates of examples in the visualization
        self.example_c = [[h_dist[i],pca.loading[i][0]] for i in range(len(examples))]
        # coordinates of the basis vectors in the visualization
        self.basis_c = [[basis_dist[i],nbasis[i][0]] for i in range(len(coeffs))]
        
        for j in range(2,dimensions):
            for i in range(len(examples)):
                self.example_c[i].append(pca.loading[i][j-1])
            for i in range(len(coeffs)):
                self.basis_c[i].append(nbasis[i][j-1])


if __name__== "__main__":
    import orngLR_Jakulin, orngSVM, orngMultiClass
    t = orange.ExampleTable('x_cmc.tab')
    #t = orange.ExampleTable('test.tab')

    def printmodel(t,c,printexamples=1):
        m = Visualizer(t,c,buckets=3)
        print "Linear model:"
        print t.domain.classVar.name,':',m.beta
        j = 0
        for i in range(len(m.coeff_names)):
            print m.coeff_names[i][0],':'
            for x in m.coeff_names[i][1:]:
                print '\t',x,':',m.coeffs[j]
                j += 1

        print "\nbasis vectors:"
        j = 0
        for i in range(len(m.coeff_names)):
            print m.coeff_names[i][0],':'
            for x in m.coeff_names[i][1:]:
                print '\t',x,':',m.basis_c[j]
                j += 1

        if printexamples:
            print "\nexamples:"
            for i in range(len(t)):
                print t[i],'->',m.example_c[i], c(t[i],orange.GetBoth), m.probfunc(m.example_c[i][0])

        print "\nprobability:"
        print "-0.5:",m.probfunc(-0.5)
        print " 0.0:",m.probfunc(0.0)
        print "+0.5:",m.probfunc(+0.5)

    print "NAIVE BAYES"
    print "==========="
    bl = orange.BayesLearner()
    bl.estimatorConstructor = orange.ProbabilityEstimatorConstructor_Laplace()
    # prevent too many estimation points
    # increase the smoothing level
    bl.conditionalEstimatorConstructorContinuous = orange.ConditionalProbabilityEstimatorConstructor_loess(windowProportion=0.5,nPoints = 10)
    c = bl(t)    
    printmodel(t,c,printexamples=0)

    print "\n\nLOGISTIC REGRESSION"
    print     "==================="
    c = orngLR_Jakulin.BasicLogisticLearner()(t)
    printmodel(t,c,printexamples=0)

    print "\n\nLINEAR SVM"
    print     "=========="
    l = orngSVM.BasicSVMLearner()
    l.kernel = 0 # linear SVM
    c = l(t)
    printmodel(t,c,printexamples=0)

    print "\n\nMARGIN SVM"
    print     "=========="
    l = orngSVM.BasicSVMLearner()
    l.kernel = 0 # linear SVM
    c = orngLR_Jakulin.MarginMetaLearner(l,folds = 1)(t)
    printmodel(t,c,printexamples=0)