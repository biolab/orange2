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


class _parse:
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

    def __call__(self,classifier,examples):
        # todo todo - support for loess
        
        beta = -self._safeRatio(classifier.distribution[1],classifier.distribution[0])
        coeffs = []
        coeff_names = []
        offsets = []
        for i in range(len(examples.domain.attributes)):
            offsets.append(len(coeff_names))
            for j in range(len(examples.domain.attributes[i].values)):
                coeff_names.append('%s = %s'%(examples.domain.attributes[i].name,examples.domain.attributes[i].values[j]))
                p1 = classifier.conditionalDistributions[i][j][1]
                p0 = classifier.conditionalDistributions[i][j][0]
                coeffs.append(self._safeRatio(p1,p0)+beta)

        # create the basis vectors for each attribute
        basis = Numeric.identity((len(coeffs)), Numeric.Float)

        # create the example matrix (only attributes)
        m = Numeric.zeros((len(examples),len(coeffs)), Numeric.Float)
        for i in range(len(examples)):
            for j in range(len(examples.domain.attributes)):
                m[i][offsets[j]+int(examples[i][j])] = 1.0
                
        return (beta, coeffs, coeff_names, basis, m, lambda x:math.exp(x)/(1.0+math.exp(x)))

class _parseLR(_parse):
    def __call__(self,classifier,examples):
        # skip domain translation and masking
        robustc = classifier.classifier
        primitivec = robustc.classifier
        beta = -primitivec.beta[0]
        coeffs = []
        coeff_names = []

        for i in classifier.translator.trans:
            coeff_names += i.description()
        # include robust LR's masking
        coeff_names = robustc.translate(coeff_names)

        for k in xrange(len(coeff_names)):
            coeffs.append(primitivec.beta[k+1])
        assert(len(coeffs)==len(coeff_names))        

        # create the basis vectors for each attribute
        basis = Numeric.identity((len(coeffs)), Numeric.Float)

        # create the example matrix (only attributes)
        m = []
        for i in range(len(examples)):
            m.append(robustc.translate(classifier.translator.extransform(examples[i])))
        m = Numeric.array(m, Numeric.Float)

        return (beta, coeffs, coeff_names, basis, m, lambda x:math.exp(x)/(1.0+math.exp(x)))


def _treshold(x):
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return 0.0
    return 0.5

class _parseSVM(_parse):
    def __call__(self,classifier,examples):
        if classifier.model['kernel_type'] != 0:
            raise "Use SVM with a linear kernel."
        if classifier.model["svm_type"] != 0:
            raise "Use ordinary SVM classification."
        if classifier.model["nr_class"] != 2:
            raise "This is not SVM with a binary class."

        coeff_names = []

        sections = []
        for i in classifier.translate.trans:
            v = i.description()
            if len(v) > 1:
                sections.append((len(coeff_names),len(v)))
            coeff_names += v

        # create the basis vectors for each attribute
        basis = Numeric.identity((len(coeff_names)), Numeric.Float)
        for (i,l) in sections:
            for j in xrange(1,l):
                for k in xrange(j):
                    basis[i+j][i+k] = -1.0
                    basis[i+k][i+j] = -1.0

        # create the example matrix (only attributes)
        m = []
        for i in range(len(examples)):
            m.append(classifier.translate.extransform(examples[i]))
        m = Numeric.array(m, Numeric.Float)

        # COMPUTE THE coefficients...
        
        beta = classifier.model["rho"][0]
        coeffs = [0.0]*len(coeff_names)
        svs = classifier.model["SV"]
        for i in xrange(classifier.model["total_sv"]):
            csv = svs[i]
            coef = csv[0][0]
            assert(len(coeffs)+1 == len(csv))
            for j in xrange(len(coeffs)):
                assert(csv[j+1][0]-1 == j)
                coeffs[j] += coef*csv[j+1][1]

        # reverse the betas if the labels got switched
        if classifier.model["label"][0] == 0:
            beta = -beta
            coeffs = [-x for x in coeffs]

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

    def __call__(self,classifier,examples):
        (beta, coeffs, coeff_names, basis, m, _probfunc) = self.parser(classifier.classifier,examples)
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
    
    def __init__(self, examples, classifier, dimensions = 2):
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

        (beta, coeffs, coeff_names, basis, m, probfunc) = parser(classifier,examples)
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
        
        if len(basis) > len(examples):
            raise "Too few examples for that many attribute values. This can be handled, ask Aleks to eliminate null spaces."

        # project the example matrix on the separating hyperplane, removing the displacement
        h_dist = Numeric.dot(m,n)
        h_proj = m - Numeric.dot(Numeric.reshape(h_dist,(len(examples),1)),Numeric.reshape(n,(1,len(coeffs))))
        h_dist -= beta # correct distance

        basis_dist = Numeric.dot(basis,n)
        basis_proj = basis - Numeric.dot(Numeric.reshape(basis_dist,(len(coeffs),1)),Numeric.reshape(n,(1,len(coeffs))))
        basis_dist -= beta

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
        m = Visualizer(t,c)
        print "Linear model:"
        print t.domain.classVar.name,':',m.beta
        for i in range(len(m.coeff_names)):
            print m.coeff_names[i],':',m.coeffs[i]

        print "\nbasis vectors:"
        for i in range(len(m.coeff_names)):
            print m.coeff_names[i],':',m.basis_c[i]

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
    c = orange.BayesLearner(t)
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