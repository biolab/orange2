#
# Module Orange Linear Model Visualization
# ----------------------------------------
#
# CVS Status: $Id$
#
# Author: Aleks Jakulin (jakulin@acm.org)
# (Copyright (C)2004 Aleks Jakulin)
#
# Purpose: Visualize all linear models (NB, LogReg, linear SVM, perceptron, etc.).
#
# ChangeLog:
#   - 2004/05/29: no more normalization, no more flipping for SVM, corrected w/r continuous attributes
#   - 2003/11/17: project initiated

import orange, orngDimRed
import math, numpy
import numpy.linalg as LinearAlgebra


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
        l.sort()
        m = len(l)/buckets
        if m == 0:
            buckets = 1
        sets = [l[:m]]
        for x in range(1,buckets):
            sets.append(l[m*x:m*(x+1)])
        values = [_avg(x) for x in sets]
        if getbuckets:
            return (values,sets)
        else:
            return values

    def quantize(self,values,x):
        pv = values[0]
        pi = 0
        for k in xrange(1,len(values)):
            if x <= values[k]:
                if values[k]-x < x-pv:
                    return k
                return pi
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
        for i in xrange(len(examples.domain.attributes)):
            for j in xrange(len(examples)):
                if examples[j][i].isSpecial():
                    raise "A missing value found in instance %d, attribute %s. Missing values are not allowed."%(j,examples.domain.attributes[i].name)

        beta = -self._safeRatio(classifier.distribution[1],classifier.distribution[0])
        coeffs = []
        coeff_names = []
        offsets = []
        setss = []
        runi = 0
        transfvalues = [] # true conditional probability for each continuous value that appears in the data
        for i in range(len(classifier.domain.attributes)):
            tc = ["%s"%classifier.domain.attributes[i].name]
            offsets.append(runi)
            LUT = {}
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
                (val,sets) = self.bucketize(examples,classifier.domain.attributes[i],buckets,1)
                p1 = 0.0
                p0 = 0.0
                # fill in the sum of probabilities, finding the nearest probability
                l = 0
                pp = -1e200
                pv = -1e200
                mm = classifier.conditionalDistributions[i].items()
                for j in xrange(len(val)):
#                    print sets[j][0],sets[j][-1],l
                    tc.append(val[j])
                    k = 0
                    while k < len(sets[j]) and l < len(mm):
                        # find the two nearest points to the given example value
                        if l < len(mm):
                            while mm[l][0] <= sets[j][k]:
                                pv = mm[l][0]
                                pp = mm[l][1]
                                l += 1
                                if l >= len(mm):
                                    break
                        # mark example values that are closer to the previous point
                        if k < len(sets[j]) and l < len(mm):
                            while sets[j][k]-pv <= mm[l][0]-sets[j][k]:
                                LUT[sets[j][k]] = self._safeRatio(pp[1],pp[0])+beta
                                p1 += pp[1]
                                p0 += pp[0]
                                k += 1
                                if k >= len(sets[j]):
                                    break
                    while k < len(sets[j]):
                        LUT[sets[j][k]] = self._safeRatio(pp[1],pp[0])+beta
                        p1 += pp[1]
                        p0 += pp[0]
                        k += 1
                    #print l, k, pv, mm[l][0], sets[j][k], len(mm), len(sets[j])
                    coeffs.append(self._safeRatio(p1,p0)+beta)
                    runi += 1
                setss.append(sets)
            else:
                raise "unknown attribute type"
            coeff_names.append(tc)
            transfvalues.append(LUT)


        # create the basis vectors for each attribute
        basis = numpy.identity((len(coeffs)), numpy.float)
        for j in range(len(classifier.domain.attributes)):
            if classifier.domain.attributes[j].varType == 2:
                for k in xrange(1,len(coeff_names[j])):
                    i = offsets[j]+k-1
                    basis[i][i] = coeff_names[j][k]

        # create the example matrix (only attributes)
        m = numpy.zeros((len(examples),len(coeffs)), numpy.float)
        for i in range(len(examples)):
            for j in range(len(classifier.domain.attributes)):
                if classifier.domain.attributes[j].varType == 1:
                    for k in coeff_names[j][1:]:
                        if not examples[i][classifier.domain.attributes[j]].isSpecial():
                            try:
                                m[i][offsets[j]+int(examples[i][classifier.domain.attributes[j]])] = 1.0
                            except:
                                print "**",examples[i][classifier.domain.attributes[j]]
                else:
                    # quantize
                    if not examples[i][classifier.domain.attributes[j]].isSpecial():
                        cv = float(examples[i][classifier.domain.attributes[j]]) # obtain the attribute value
                    else:
                        cv = 0.0
                    k = self.quantize(coeff_names[j][1:],cv) # obtain the right bucket
                    tm = transfvalues[j][cv]  # true margin
                    mu = coeffs[offsets[j]+k] # multiplier for the bucket
                    if abs(mu)>1e-6:
                        ac = tm/mu
                    else:
                        ac = tm
                    m[i][offsets[j]+k] = ac

        return (beta, coeffs, coeff_names, basis, m, lambda x:math.exp(x)/(1.0+math.exp(x)))

class _parseLR(_parse):
    def getDescriptors(self, translator, examples, buckets):
        tcoeff_names = []
        descriptors = [] # used for managing continuous atts
        proto_example = orange.Example(examples[0]) # used for converting bucket averages into realistic values
        true_values = []
        for i in range(len(translator.trans)):
            t = translator.trans[i]
            tc = ["%s"%t.attr.name]
            tv = []
            d = t.description()
            if d[0]==0:
                # continuous
                values = self.bucketize(examples, t.attr, buckets)
                tc += values
                descriptors.append((i,-1))
                for v in values:
                    proto_example[t.attr] = v
                    tp = translator.extransform(proto_example)
                    tv.append(tp[t.idx])
            else:
                # nominal
                x = 0
                for n in d[2]:
                    if n!='':
                        tc.append(n)
                        descriptors.append((i,x))
                        x += 1
            true_values.append(tv)
            tcoeff_names.append(tc)
        return descriptors, tcoeff_names, true_values

    def getNames(self, descriptors, tcoeff_names, true_values):
        # filter the coeff_names using these masked descriptors
        coeff_names = []
        cur_i = -1
        contins = []
        tr_values = []
        total = 0
        for (a,b) in descriptors:
            if cur_i == a:
                coeff_names[-1].append(tcoeff_names[a][b+1])
                total += 1
            else:
                tr_values.append(true_values[a])
                if b == -1:
                    # continuous
                    contins.append(len(coeff_names))
                    coeff_names.append(tcoeff_names[a])
                    total += len(coeff_names[-1]) - 1
                else:
                    coeff_names.append([tcoeff_names[a][0],tcoeff_names[a][1+b]])
                    total += 1
                    cur_i = a
        return coeff_names,total,contins,tr_values

    def getBasis(self, total,xcoeffs,contins, coeff_names, tr_values):
        # create the basis vectors for each attribute
        basis = numpy.identity((total), numpy.float)

        # fix up the continuous attributes in coeffs (duplicate) and in basis (1.0 -> value[i])
        x = numpy.ones((total,), numpy.float)
        coeffs = []
        lookup = []
        conti = 0
        j = 0
        jj = 0
        contins.append(-1) # sentry
        nlookup = []
        for i in xrange(len(coeff_names)):
            nlookup.append(len(coeffs))
            if i==contins[conti]:
                # continuous
                conti += 1
                lookup.append(len(coeffs))
                for k in xrange(len(coeff_names[i])-1):
                    coeffs.append(xcoeffs[jj])
                    v = tr_values[i][k]
                    x[j] = v
                    j += 1
                jj += 1
            else:
                # discrete
                for k in xrange(len(coeff_names[i])-1):
                    lookup.append(len(coeffs))
                    coeffs.append(xcoeffs[jj])
                    j += 1
                    jj += 1
        basis *= x
        return (basis,lookup,nlookup,coeffs)

    def getExamples(self,exx,tex,dim,classifier,lookup,nlookup,contins, coeff_names):
        # create the example matrix (only attributes)
        m = numpy.zeros((len(tex),dim), numpy.float)
        for j in xrange(len(tex)):
            tv = tex[j]
            # do the insertion and quantization
            # copy all coefficients
            for i in xrange(len(lookup)):
                try:
                    m[j][lookup[i]] = tv[i]
                except:
                    m[j][lookup[i]] = 0 # missing value

            # quantize the continuous attributes
            for i in contins[:-1]:
                # obtain the marker
                vals = coeff_names[i][1:]
                x = float(exx[j][coeff_names[i][0]]) # get the untransformed value of the attribute
                newi = self.quantize(vals,x)
                # move the value to the right bucket
                v = m[j][nlookup[i]]
                m[j][nlookup[i]] = 0.0
                m[j][nlookup[i]+newi] = v
        return m

    def __call__(self,classifier,examples, buckets):
        # skip domain translation and masking
        robustc = classifier.classifier
        primitivec = robustc.classifier
        beta = -primitivec.beta[0]

        (descriptors,prevnames,trans_values) = self.getDescriptors(classifier.translator,examples,buckets)

        # include robust LR's masking
        descriptors = robustc.translate(descriptors)

        (coeff_names, total, contins, tr_values) = self.getNames(descriptors, prevnames, trans_values)

        xcoeffs = primitivec.beta[1:]

        (basis,lookup,nlookup,coeffs) = self.getBasis(total, xcoeffs, contins, coeff_names, tr_values)

        tex = []
        for ex in examples:
            tex.append(robustc.translate(classifier.translator.extransform(ex)))

        m = self.getExamples(examples,tex,len(basis),classifier,lookup,nlookup,contins, coeff_names)

        return (beta, coeffs, coeff_names, basis, m, lambda x:math.exp(x)/(1.0+math.exp(x)))


class _parseSVM(_parseLR):
    def __call__(self,classifier,examples, buckets):
        if classifier.model['kernel_type'] != 0:
            raise "Use SVM with a linear kernel."
        if classifier.model["svm_type"] != 0:
            raise "Use ordinary SVM classification."
        if classifier.model["nr_class"] != 2:
            raise "This is not SVM with a binary class."

        (descriptors,prevnames,trans_values) = self.getDescriptors(classifier.translate,examples,buckets)

        (coeff_names, total, contins, tr_values) = self.getNames(descriptors, prevnames, trans_values)

        beta = classifier.beta
        xcoeffs = classifier.xcoeffs

        (basis,lookup,nlookup,coeffs) = self.getBasis(total, xcoeffs, contins, coeff_names, tr_values)

        tex = []
        for i in range(len(examples)):
            tex.append(classifier.translate.extransform(examples[i]))

        m = self.getExamples(examples,tex,len(basis),classifier,lookup,nlookup,contins, coeff_names)

        return (beta, coeffs, coeff_names, basis, m, _treshold)


class _marginConverter:
    def __init__(self,coeff,estdomain,estimator):
        self.coeff = coeff
        self.estdomain = estdomain
        self.cv = self.estdomain.classVar(0)
        self.estimator = estimator

    def __call__(self, r):
        ex = orange.Example(self.estdomain,[r*self.coeff,self.cv]) # need a dummy class value
        p = self.estimator(ex,orange.GetProbabilities)
        return p[1]

class _parseMargin(_parse):
    def __init__(self,marginc,parser):
        self.parser = parser
        self.marginc = marginc

    def __call__(self,classifier,examples, buckets):
        (beta, coeffs, coeff_names, basis, m, _probfunc) = self.parser(classifier.classifier,examples, buckets)
        return (beta, coeffs, coeff_names, basis, m, _marginConverter(self.marginc.coeff, self.marginc.estdomain, self.marginc.estimator))


class Visualizer:
    def findParser(self, classifier):
        if type(classifier)==orange.BayesClassifier:
             return _parseNB()
        else:
            try:
                name = classifier._name
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

    def __init__(self, examples, classifier, dimensions = 2, buckets = 3, getpies = 0, getexamples = 1):
        # error detection
        if len(examples.domain.classVar.values) != 2:
            raise "The domain does not have a binary class. Binary class is required."

        all_attributes = [i for i in examples.domain.attributes]+[examples.domain.classVar]

        # acquire the linear model
        parser = self.findParser(classifier)

        (beta, coeffs, coeff_names, basis, m, probfunc) = parser(classifier,examples, buckets)
        #print "examples:"
        #print m
        #print "basis:"
        #print basis
        self.basis = basis
        self.m = m

        # get the parameters of the hyperplane, and normalize it
        n = numpy.array(coeffs, numpy.float)
        #length = numpy.sqrt(numpy.dot(n,n))
        #ilength = 1.0/length
        #n *= ilength
        #beta = ilength*xbeta
        #self.probfunc = lambda x:probfunc(x*length)
        self.probfunc = lambda x:probfunc(x)

        if getexamples or getpies or dimensions > 1:
            # project the example matrix on the separating hyperplane, removing the displacement
            h_dist = numpy.dot(m,n)
            h_proj = m - numpy.dot(numpy.reshape(h_dist,(len(examples),1)),numpy.reshape(n,(1,len(coeffs))))/numpy.dot(n,n)
            h_dist -= beta # correct distance

        # perform classification for all examples
        if getpies:
            self.pies = []
            for j in range(len(examples)):
                p1 = self.probfunc(h_dist[j])
                p0 = 1-p1
                t = []
                # calculate for all coefficients, prior (beta) is last
                projj = m[j]*n # projection in the space defined by the basis and the hyperplane
                tn = numpy.concatenate((projj,[beta-h_dist[j]]))
                evidence0 = -numpy.clip(tn,-1e200,0.0)
                evidence1 = numpy.clip(tn,0.0,1e200)
                # evidence for label 0
                evidence0 *= p0/max(numpy.sum(evidence0),1e-6)
                # evidence for label 1
                evidence1 *= p1/max(numpy.sum(evidence1),1e-6)
                self.pies.append((evidence0,evidence1,projj))

        basis_dist = numpy.dot(basis,n)
        basis_proj = basis - numpy.dot(numpy.reshape(basis_dist,(len(coeffs),1)),numpy.reshape(n,(1,len(coeffs))))/numpy.dot(n,n)
        basis_dist -= beta

        # coordinates of the basis vectors in the visualization
        self.basis_c = [[basis_dist[i]] for i in range(len(coeffs))]

        self.coeff_names = coeff_names
        self.coeffs = coeffs
        self.beta = beta

        if getexamples:
            # coordinates of examples in the visualization
            self.example_c = [[h_dist[i]] for i in range(len(examples))]

        if dimensions > 1:
            # perform standardization of attributes; the pa
            for i in xrange(len(coeffs)):
                # standardize the examples and return parameters
                (h_proj[:,i],transf) = orngDimRed.VarianceScaling(h_proj[:,i])
                # now transform the basis and the examples with the same parameters, so that nothing changes
                (basis_proj[:,i],transf) = orngDimRed.VarianceScaling(basis_proj[:,i],transf)
                #(m[:,i],transf) = orngDimRed.VarianceScaling(m[:,i],transf)

            # obtain the second dimension using PCA
            pca = orngDimRed.PCA(h_proj,dimensions-1)
            # now transform the basis using the same matrix (need N)
            # U' = U*D*V
            # N' * V^-1 * D^-1 = N
            # N = N' * (D*V)^-1
            DV = numpy.dot(numpy.identity(len(coeffs),numpy.float)*numpy.clip(pca.variance,1e-6,1e6),pca.factors)
            nbasis = numpy.dot(basis_proj,LinearAlgebra.inv(DV))

            for j in range(dimensions-1):
                for i in range(len(coeffs)):
                    self.basis_c[i].append(nbasis[i][j])
            if getexamples:
                for j in range(dimensions-1):
                    for i in range(len(examples)):
                        self.example_c[i].append(pca.loading[i][j])


if __name__== "__main__":
    import orngLR_Jakulin, orngSVM, orngMultiClass

    def printmodel(t,c,printexamples=1):
        m = Visualizer(t,c,buckets=3,getpies=1)
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

        idx = 0
        print "\npie for example",idx,':',t[idx]
        (e0,e1,vector) = m.pies[idx]
        def printpie(e,p):
            x = 0
            for i in range(len(m.coeff_names)):
                for j in m.coeff_names[i][1:]:
                    if e[x] > 0.001:
                        print '\t%2.1f%% : '%(100*e[x]),m.coeff_names[i][0],'=',
                        if type(j)==type(1.0):
                            print t[idx][m.coeff_names[i][0]] # continuous
                        else:
                            print j # discrete
                    x += 1
            if e[x] > 0.001:
                print '\t%2.1f%% : '%(100*e[x]),"BASELINE"
        print "Attributes in favor of %s = %s [%f]"%(t.domain.classVar.name,t.domain.classVar.values[0],1-m.probfunc(m.example_c[idx][0]))
        printpie(e0,1-m.probfunc(m.example_c[idx][0]))
        print "Attributes in favor of %s = %s [%f]"%(t.domain.classVar.name,t.domain.classVar.values[1],m.probfunc(m.example_c[idx][0]))
        printpie(e1,m.probfunc(m.example_c[idx][0]))

        print "\nProjection of the example in the basis space:"
        j = 0
        for i in range(len(m.coeff_names)):
            print m.coeff_names[i][0],':'
            for x in m.coeff_names[i][1:]:
                print '\t',x,'=',vector[j]
                j += 1
        print "beta:",-m.beta

    #t = orange.ExampleTable('c:/proj/domains/voting.tab') # discrete
    t = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\shuttle.tab" ) # discrete

    #t = orange.ExampleTable('c_cmc.tab') # continuous

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
    l.for_nomogram = 1
    c = l(t)
    printmodel(t,c,printexamples=0)

    print "\n\nMARGIN SVM"
    print     "=========="
    l = orngSVM.BasicSVMLearner()
    l.kernel = 0 # linear SVM
    l.for_nomogram = 1
    c = orngLR_Jakulin.MarginMetaLearner(l,folds = 1)(t)
    printmodel(t,c,printexamples=0)
