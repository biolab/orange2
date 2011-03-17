from string import join

import numpy
from numpy.linalg import inv
from numpy import dot, sqrt

import Orange
import statc


########################################################################
# Linear Regression

class LinearRegressionLearner(object):
    def __new__(self, data=None, name='linear regression', **kwds):
        learner = object.__new__(self, **kwds)
        if data:
            learner.__init__(name,**kwds) # force init
            return learner(data)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='linear regression', beta0=True,
                 use_attributes=None, stepwise=False, add_sig=0.05,
                 remove_sig=0.2, stepwise_before=True, **kw):
        self.name = name
        self.beta0 = beta0
        self.stepwise = stepwise
        self.stepwise_before = stepwise_before
        self.add_sig = add_sig
        self.remove_sig = remove_sig
        self.use_attributes = use_attributes
        self.__dict__.update(kw)

    def __call__(self, data, weight=None):
        if not self.use_attributes == None:
            new_domain = Orange.data.Domain(self.use_attributes,
                                            data.domain.classVar)
            new_domain.addmetas(data.domain.getmetas())
            data = Orange.data.Table(new_domain, data)
            
        if self.stepwise and self.stepwise_before:
            use_attributes=stepwise(data, add_sig=self.add_sig,
                                    remove_sig=self.remove_sig)
            new_domain = Orange.data.Domain(use_attributes, data.domain.classVar)
            new_domain.addmetas(data.domain.getmetas())
            data = Orange.data.Table(new_domain, data)

        # continuization (replaces discrete with continuous attributes)
        continuizer = Orange.feature.continuization.DomainContinuizer()
        continuizer.multinomialTreatment = continuizer.FrequentIsBase
        continuizer.zeroBased = True
        domain0 = continuizer(data)
        data = data.translate(domain0)

        if self.stepwise and not self.stepwise_before:
            use_attributes = stepwise(data, weight, add_sig=self.add_sig,
                                      remove_sig=self.remove_sig)
            new_domain = Orange.data.Domain(use_attributes, data.domain.classVar)
            new_domain.addmetas(data.domain.getmetas())
            data = Orange.data.Table(new_domain, data)        
        
        # missing values handling (impute missing)
        imputer = Orange.feature.imputation.ImputerConstructor_model()
        imputer.learnerContinuous = Orange.regression.mean.MeanLearner()
        imputer.learnerDiscrete = Orange.classification.majority.MajorityLearner()
        imputer = imputer(data)
        data = imputer(data)

        # convertion to numpy
        A, y, w = data.toNumpy()        # weights ??
        if A == None:
            n = len(data)
            m = 0
        else:
            n, m = numpy.shape(A)
     
        if self.beta0 == True:
             if A == None:
                 X = numpy.ones([len(data), 1])
             else:
                 X = numpy.insert(A, 0, 1, axis=1) # adds a column of ones
        else:
             X = A

        # set weights
        W = numpy.identity(len(data))
        if weight:
            for di, d in enumerate(data):
                W[di, di] = float(d[weight])

        # adds some robustness by computing the pseudo inverse;
        # normal inverse could fail due to singularity of the X.T * W * X
        D = dot(dot(numpy.linalg.pinv(dot(dot(X.T, W), X)), X.T), W)
        beta = dot(D, y)

        yEstimated = dot(X, beta)  # estimation
        # some desriptive statistisc
        muY, sigmaY = numpy.mean(y), numpy.std(y)
        muX, covX = numpy.mean(X, axis=0), numpy.cov(X, rowvar=0)
 
        # model statistics
        SST = numpy.sum((y - muY) ** 2)
        SSR = numpy.sum((yEstimated - muY) ** 2)
        SSE, RSquare = SST - SSR, SSR / SST
        R = numpy.sqrt(RSquare) # coefficient of determination
        RAdjusted = 1 - (1 - RSquare) * (n - 1) / (n - m - 1)
        F = (SSR / m) / (SST - SSR / (n - m - 1)) # F statistisc
        df = m - 1
 
        sigmaSquare = SSE / (n - m - 1)

        # standard error of estimated coefficients
        errCoeff = sqrt(sigmaSquare * inv(dot(X.T, X)).diagonal())
 
        # t statistisc, significance
        t = beta / errCoeff
        df = n - 2
        significance = []
        for tt in t:
            try:
                significance.append(statc.betai(df * 0.5, 0.5,
                                                df / (df + tt * tt)))
            except:
                significance.append(1.0)
 
        # standardized coefficients
        if m > 0:
            stdCoeff = (sqrt(covX.diagonal()) / sigmaY)  * beta
        else:
            stdCoeff = (sqrt(covX) / sigmaY)  * beta
 
        model = {'descriptives': { 'meanX': muX, 'covX': covX, 'meanY': muY,
                                   'sigmaY': sigmaY},
                 'model' : {'estCoeff': beta, 'stdErrorEstimation': errCoeff},
                 'model summary': {'TotalVar': SST, 'ExplVar': SSE,
                                   'ResVar': SSR, 'R': R, 'RAdjusted': RAdjusted,
                                   'F': F, 't': t, 'sig': significance}}
        return LinearRegression(statistics=model, domain=data.domain,
                                name=self.name, beta0=self.beta0, imputer=imputer)

class LinearRegression(Orange.classification.Classifier):
    def __init__(self, **kwds):
        for a, b in kwds.items():
            self.setattr(a, b)
        self.beta = self.statistics['model']['estCoeff']

    def __call__(self, example, resultType=Orange.classification.Classifier.GetValue):
        ex = Orange.data.Instance(self.domain, example)
        ex = self.imputer(ex)
        ex = numpy.array(ex.native())

        if self.beta0:
            if len(self.beta) > 1:
                yhat = self.beta[0] + dot(self.beta[1:], ex[:-1])
            else:
                yhat = self.beta[0]
        else:
            yhat = dot(self.beta, ex[:-1])
        yhat = Orange.data.Value(yhat)
         
        if resultType == Orange.classification.Classifier.GetValue:
            return yhat
        if resultType == Orange.classification.Classifier.GetProbabilities:
            return Orange.statistics.distribution.Continuous({1.0: yhat})
        return (yhat, Orange.statistics.distribution.Continuous({1.0: yhat}))
    
    def __str__(self):
        err = self.statistics['model']['stdErrorEstimation']
        t = self.statistics['model summary']['t']
        sig = self.statistics['model summary']['sig']
        
        s = ' '.join(['%10s' % l for l in
                      ('Variable', 'Coeff Est', 'Std Error', 't-value', 'p')])

        fmt = '\n%10s ' + ' '.join(["%10.3f"] * 4)
        if self.beta0 == True:
            s += fmt % ('Constant', self.beta[0], err[0], t[0], sig[0])
            for i in range(len(self.domain.attributes) - 1):
                s +=  fmt % (self.domain.attributes[i].name,
                             self.beta[i + 1], err[i + 1], t[i + 1], sig[i + 1])
        else:
            for i in range(len(self.domain.attributes) - 1):
                s +=  fmt % (self.domain.attributes[i].name,
                             self.beta[i], err[i], t[i], sig[i])
        return s

def get_sig(m1, m2, n):
    if m1 == None or m2 == None:
        return 1.0
    p1, p2 = len(m1.domain.attributes), len(m2.domain.attributes)
    RSS1 = m1.statistics["model summary"]["ExplVar"]
    RSS2 = m2.statistics["model summary"]["ExplVar"]
    if RSS1 <= RSS2 or p2 <= p1 or n <= p2 or RSS2 <= 0:
        return 1.0
    F = ((RSS1 - RSS2) / (p2 - p1)) / (RSS2 / (n - p2))
    return statc.fprob(int(p2 - p1), int(n - p2), F)

def stepwise(data, weight, add_sig=0.05, remove_sig=0.2):
    inc_atts = []
    not_inc_atts = [at for at in data.domain.attributes]

    changed_model = True
    while changed_model:
        changed_model = False
        # remove all unsignificant conditions (learn several models,
        # where each time one attribute is removed and check significance)
        orig_lin_reg = LinearRegressionLearner(data, use_attributes=inc_atts)
        reduced_lin_reg = []
        for ati in range(len(inc_atts)):
            try:
                reduced_lin_reg.append(LinearRegressionLearner(data, weight,
                        use_attributes=inc_atts[:ati] + inc_atts[(ati + 1):]))
            except:
                reduced_lin_reg.append(None)
        
        sigs = [get_sig(r, orig_lin_reg, len(data)) for r in reduced_lin_reg]
        if sigs and max(sigs) > remove_sig:
            # remove that attribute, start again
            crit_att = inc_atts[sigs.index(max(sigs))]
            not_inc_atts.append(crit_att)
            inc_atts.remove(crit_att)
            changed_model = True
            continue

        # add all significant conditions (go through all attributes in
        # not_inc_atts, is there one that significantly improves the model?
        more_complex_lin_reg = []
        for ati in range(len(not_inc_atts)):
            try:
                more_complex_lin_reg.append(LinearRegressionLearner(data,
                        weight, use_attributes=inc_atts + [not_inc_atts[ati]]))
            except:
                more_complex_lin_reg.append(None)

        sigs = [get_sig(orig_lin_reg, r, len(data)) for r in more_complex_lin_reg]
        if sigs and min(sigs) < add_sig:
            best_att = not_inc_atts[sigs.index(min(sigs))]
            inc_atts.append(best_att)
            not_inc_atts.remove(best_att)
            changed_model = True
    return inc_atts

