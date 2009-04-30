import orange
import numpy
from numpy.linalg import inv
from numpy import dot, sqrt
import statc
from string import join

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

    def __init__(self, name='linear regression', beta0 = True, stepwise=False, add_sig=0.05, remove_sig=0.2, use_attributes=None, **kw):
        self.name = name
        self.beta0 = beta0
        self.stepwise=stepwise
        self.add_sig=add_sig
        self.remove_sig=remove_sig
        self.use_attributes=use_attributes
        self.__dict__.update(kw)

    def __call__(self, data, weight=None):
        if self.stepwise:
            self.use_attributes=stepwise(data,add_sig=self.add_sig,remove_sig=self.remove_sig)
        if not self.use_attributes == None:
            new_domain = orange.Domain(self.use_attributes, data.domain.classVar)
            data = orange.ExampleTable(new_domain, data)

        # continuization (replaces discrete with continuous attributes)
        continuizer = orange.DomainContinuizer()
        continuizer.multinomialTreatment = continuizer.FrequentIsBase
        continuizer.zeroBased = True
        domain0 = continuizer(data)
        data = data.translate(domain0)
        
        # missing values handling (impute missing)
        imputer = orange.ImputerConstructor_model()
        imputer.learnerContinuous = orange.MajorityLearner()
        imputer.learnerDiscrete = orange.MajorityLearner()
        imputer = imputer(data)
        data = imputer(data)

        # convertion to numpy
        A, y, w = data.toNumpy()        # weights ??
        if A==None:
            n = len(data)
            m = 0
        else:
            n, m = numpy.shape(A)
     
        if self.beta0 == True:
             if A==None:
                 X = numpy.ones([len(data),1])
             else:
                 X = numpy.insert(A,0,1,axis=1) # adds a column of ones
        else:
             X = A
             
        beta, resid, rank, s = numpy.linalg.lstsq(X,y)  # should also check for rank
        
        yEstimated = dot(X,beta)  # estimation
        # some desriptive statistisc
        muY, sigmaY = numpy.mean(y), numpy.std(y)
        muX, covX = numpy.mean(X, axis = 0), numpy.cov(X, rowvar = 0)
 
        # model statistics
        SST, SSR = numpy.sum((y - muY) ** 2), numpy.sum((yEstimated - muY) ** 2)
        SSE, RSquare = SST-SSR, SSR/SST
        R = numpy.sqrt(RSquare) # coefficient of determination
        RAdjusted = 1 - (1 - RSquare) * (n - 1) / (n - m - 1)
        F = (SSR / m) / (SST - SSR / (n - m - 1)) # F statistisc
        df = m - 1
 
        sigmaSquare = SSE / (n-m-1)

        # standard error of estimated coefficients
        errCoeff = sqrt(sigmaSquare * inv(dot(X.T,X)).diagonal())
 
        # t statistisc, significance
        t = beta / errCoeff
        df = n-2
        significance = []
        for tt in t:
            try:
                significance.append(statc.betai(df*0.5,0.5,df/(df+tt*tt)))
            except:
                significance.append(1.0)
 
        # standardized coefficients
        if m>0:
            stdCoeff = (sqrt(covX.diagonal()) / sigmaY)  * beta
        else:
            stdCoeff = (sqrt(covX) / sigmaY)  * beta
 
        model = {'descriptives': { 'meanX' : muX, 'covX' : covX, 'meanY' : muY, 'sigmaY' : sigmaY},
                 'model' : {'estCoeff' : beta, 'stdErrorEstimation': errCoeff},
                 'model summary': {'TotalVar' : SST, 'ExplVar' : SSE, 'ResVar' : SSR, 'R' : R, 'RAdjusted' : RAdjusted,
                                   'F' : F, 't' : t, 'sig': significance}}
 
        return LinearRegression(statistics = model, domain = data.domain, name = self.name, beta0 = self.beta0, imputer=imputer)

class LinearRegression(orange.Classifier):
    def __init__(self, **kwds):
        for a,b in kwds.items():
            self.setattr(a,b)
        self.beta = self.statistics['model']['estCoeff']

    def __call__(self, example, resultType = orange.GetValue):
        ex = orange.Example(self.domain, example)
        ex = self.imputer(ex)
        ex = numpy.array(ex.native())

        if self.beta0:
            if len(self.beta)>1:
                yhat = self.beta[0] + dot(self.beta[1:], ex[:-1])
            else:
                yhat = self.beta[0]
        else:
            yhat = dot(self.beta, ex[:-1])
        yhat = orange.Value(yhat)
         
        if resultType == orange.GetValue:
            return yhat
        if resultType == orange.GetProbabilities:
            return orange.ContDistribution({1.0: yhat})
        return (yhat, orange.ContDistribution({1.0: yhat}))


def printLinearRegression(lr):
    """pretty-prints linear regression model"""
    beta = lr.beta
    err = lr.statistics['model']['stdErrorEstimation']
    t = lr.statistics['model summary']['t']
    sig = lr.statistics['model summary']['sig'] 
    beta0 = lr.beta0
    
    labels = ('Variable', 'Coeff Est', 'Std Error', 't-value', 'p')
    print join(['%10s' % l for l in labels], ' ')

    fmt = "%10s " + join(["%10.3f"]*4, " ")
    if beta0 == True:
        print fmt % ('Constant', beta[0], err[0], t[0], sig[0])
        for i in range(len(lr.domain.attributes)-1):
            print fmt % (lr.domain.attributes[i].name, beta[i+1], err[i+1], t[i+1], sig[i+1])
    else:
        for i in range(len(lr.domain.attributes)-1):
            print fmt % (lr.domain.attributes[i].name, beta[i], err[i], t[i], sig[i])       

def get_sig(m1, m2, n):
    if m1==None or m2==None:
        return 1.0
    p1, p2 = len(m1.domain.attributes), len(m2.domain.attributes)
    RSS1, RSS2 = m1.statistics["model summary"]["ExplVar"], m2.statistics["model summary"]["ExplVar"]
    if RSS1<=RSS2 or p2<=p1 or n<=p2:
        return 1.0
    F = ((RSS1 - RSS2)/(p2-p1))/(RSS2/(n-p2))
    return statc.fprob(int(p2-p1),int(n-p2),F)

def stepwise(data, add_sig = 0.05, remove_sig = 0.2):
    inc_atts = []
    not_inc_atts = [at for at in data.domain.attributes]

    changed_model = True
    while changed_model:
        changed_model = False
        # remove all unsignificant conditions (learn several models, where each time 
        # one attribute is removed and check significance)
        orig_lin_reg = LinearRegressionLearner(data, use_attributes = inc_atts)
        reduced_lin_reg = []
        for ati in range(len(inc_atts)):
            try:
                reduced_lin_reg.append(LinearRegressionLearner(data, use_attributes = inc_atts[:ati]+inc_atts[(ati+1):]))
            except:
                reduced_lin_reg.append(None)
                           
        sigs = [get_sig(r,orig_lin_reg,len(data)) for r in reduced_lin_reg]
        if sigs and max(sigs) > remove_sig:
            # remove that attribute, start again
            crit_att = inc_atts[sigs.index(max(sigs))]
            not_inc_atts.append(crit_att)
            inc_atts.remove(crit_att)
            changed_model = True
            continue

        # add all significant conditions (go through all attributes in not_inc_atts,
        # is there one that significantly improves the model?
        more_complex_lin_reg = []
        for ati in range(len(not_inc_atts)):
            try:
                more_complex_lin_reg.append(LinearRegressionLearner(data, use_attributes = inc_atts + [not_inc_atts[ati]]))
            except:
                more_complex_lin_reg.append(None)        
        sigs = [get_sig(orig_lin_reg,r,len(data)) for r in more_complex_lin_reg]
        if sigs and min(sigs) < add_sig:
            best_att = not_inc_atts[sigs.index(min(sigs))]
            inc_atts.append(best_att)
            not_inc_atts.remove(best_att)
            changed_model = True
    return inc_atts        

    
########################################################################
# Partial Least-Squares Regression (PLS)

# Function is used in PLSRegression
def standardize(matrix):
    mean = numpy.mean(matrix, axis = 0)
    one = numpy.ones(numpy.shape(matrix))
    std = numpy.std(matrix, axis = 0)
    return (matrix - numpy.multiply(mean, one))/std

# Function is used in PLSRegression
def normalize(vector):
    return vector / numpy.linalg.norm(vector)

class PLSRegressionLearner(object):
    """PLSRegressionLearner(data, y, x=None, nc=None)"""
    def __new__(self, data=None, name='PLS regression', **kwds):
        learner = object.__new__(self, **kwds)
        if data:
            learner.__init__(name) # force init
            return learner(data)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='PLS regression', nc = None):
        self.name = name
        self.nc = nc

    def __call__(self, data, y, x=None, nc=None, weight=None):
        if x == None:
            x = [v for v in data.domain.variables if v not in y]

        dataX = data.select(x)
        dataY = data.select(y)
        print y, dataY
        
        # transformation to numpy arrays
        X = dataX.toNumpy()[0]
        Y = dataY.toNumpy()[0]
    
        # data dimensions
        n, mx = numpy.shape(X)
        my = numpy.shape(Y)[1]

        # Z-scores of original matrices
        YMean = numpy.mean(Y, axis = 0)
        YStd = numpy.std(Y, axis = 0)
        XMean = numpy.mean(X, axis = 0)
        XStd = numpy.std(X, axis = 0) 
        X,Y = standardize(X), standardize(Y)

        P = numpy.empty((mx,Ncomp))
        C = numpy.empty((my,Ncomp))
        T = numpy.empty((n,Ncomp))
        U = numpy.empty((n,Ncomp))
        B = numpy.zeros((Ncomp,Ncomp))
        W = numpy.empty((mx,Ncomp))
        E,F = X,Y
    
        # main algorithm
        for i in range(Ncomp):
            u = numpy.random.random_sample((n,1))
            w = normalize(dot(E.T,u))
            t = normalize(dot(E,w))
            dif = t    
            # iterations for loading vector t
            while numpy.linalg.norm(dif) > 10e-16:
                c = normalize(dot(F.T,t))
                u = dot(F,c)
                w = normalize(dot(E.T,u))
                t0 = normalize(dot(E,w))
                dif = t - t0
                t = t0
    
            T[:,i] = t.T
            U[:,i] = u.T
            C[:,i] = c.T
            W[:,i] = w.T
            b = dot(t.T,u)[0]
            B[i][i] = b
            p = dot(E.T,t)
            P[:,i] = p.T
            E = E - dot(t,p.T)
            F = F - b * dot(t,c.T)
    
        # esimated Y
        YE = dot(dot(T,B),C.T)*YStd + YMean
        Y = Y*numpy.std(Y, axis = 0)+ YMean
        BPls = dot(dot(numpy.linalg.pinv(P.T),B),C.T)    
        return PLSRegression(domain=data.domain, BPls=BPls, YMean=YMean, YStd=YStd, XMean=XMean, XStd=XStd, name=self.name)

class PLSRegression:
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, example):
       ex = orange.Example(self.domain, example)
       ex = numpy.array(ex.native())
       ex = (ex - self.XMean) / self.XStd
       yhat = dot(ex, self.BPls) * self.YStd + self.YMean        
       return yhat
