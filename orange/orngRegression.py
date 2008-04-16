import orange
import numpy
from numpy.linalg import inv
from numpy import dot, sqrt
import statc
from string import join

verbose = 0

########################################################################
# Linear Regression

class LinearRegressionLearner(object):
    def __new__(self, data=None, name='linear regression', **kwds):
        learner = object.__new__(self, **kwds)
        if data:
            learner.__init__(name) # force init
            return learner(data)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='linear regression', intercept=True):
        self.name = name
        self.intercept = intercept

    def __call__(self, data, weight=None, imputer=None, continuizer=None):

        # missing values handling (impute missing)
        if imputer == None:           
            imputer = orange.ImputerConstructor_model()
            imputer.learnerContinuous = orange.MajorityLearner()
            imputer.learnerDiscrete = orange.BayesLearner()
            imputer = imputer(data)
        data = imputer(data)
 
        # continuization (replaces discrete with continuous attributes)
        if continuizer == None:
            continuizer = orange.DomainContinuizer()
            continuizer.multinomialTreatment = continuizer.FrequentIsBase
            continuizer.zeroBased = True
            domain0 = continuizer(data)
        data = data.translate(domain0)
 
        #   convertion to numpy
        A, y, w = data.toNumpy()        
        n, m = numpy.shape(A)
     
        if self.intercept == True:
             X = numpy.insert(A,0,1,axis=1) # adds a column of ones
        else:
             X = A
             
        self.beta, self.resid, rank, s = numpy.linalg.lstsq(X,y)  
        
        yhat = dot(X, self.beta)  # estimation
 
        # some desriptive statistisc
        self.muY, self.sigmaY = numpy.mean(y), numpy.std(y)
        self.muX, self.covX = numpy.mean(A, axis = 0), numpy.cov(A, rowvar = 0)
 
        # model statistics
        self.SST, self.SSR = numpy.sum((y - self.muY) ** 2), numpy.sum((yhat - self.muY) ** 2)
        self.SSE, self.RSquare = self.SST - self.SSR, self.SSR / self.SST
        self.R = numpy.sqrt(self.RSquare) # coefficient of determination
        self.RAdjusted = 1 - (1 - self.RSquare) * (n - 1) / (n - m - 1)
        self.F = (self.SSR / m) / (self.SST - self.SSR / (n - m - 1)) # F statistisc
        self.df = m - 1
 
        self.sigmaSquare = self.SSE / (n-m-1)

        # standard error of estimated coefficients
        self.errCoeff = sqrt(self.sigmaSquare * inv(dot(X.T,X)).diagonal())
 
        # t statistisc, significance
        self.t = self.beta / self.errCoeff
        self.significance = [statc.betai(self.df*0.5,0.5,self.df/(self.df+tt*tt)) for tt in self.t]
 
        #   standardized coefficients
        if self.intercept == True:   
             stdCoeff = (sqrt(self.covX.diagonal()) / self.sigmaY)  * self.beta[1:]
        else:
             stdCoeff = (sqrt(self.covX.diagonal()) / self.sigmaY)  * self.beta

        self.domain = data.domain
         
        return LinearRegression(model = self)
      

class LinearRegression:
    def __init__(self, **kwds):
        self.__dict__ = kwds
        
    def __call__(self, example):
        ex = orange.Example(self.model.domain, example)
        ex = numpy.array(ex.native())

        if self.model.intercept:
            yhat = self.model.beta[0] + dot(self.model.beta[1:], ex[:-1])
        else:
            yhat = dot(self.model.beta, ex[:-1])
         
        return yhat
    
    def __str__(self):

        beta = self.model.beta
        err = self.model.errCoeff
        t = self.model.t
        sig = self.model.significance 
        intercept = self.model.intercept
        domain = self.model.domain
    
        labels = ('Variable', 'Coeff Est', 'Std Error', 't-value', 'p')
        print join(['%10s' % l for l in labels], ' ')

        fmt = "%10s " + join(["%10.3f"]*4, " ")
        if intercept == True:
            print fmt % ('Constant', beta[0], err[0], t[0], sig[0])
            for i in range(len(domain.attributes)-1):
                print fmt % (domain.attributes[i].name, beta[i+1], err[i+1], t[i+1], sig[i+1])
        else:
            for i in range(len(domain.attributes)-1):
                print fmt % (domain.attributes[i].name, beta[i], err[i], t[i], sig[i])         
        
        return 'Table of some statistics'


########################################################################
#    
#   Multivariate Multiple Regression
#
#

class MVRegressionLearner(object):
    def __new__(self, data=None, y=None, x=None, name='MV regression', **kwds):
        learner = object.__new__(self, **kwds)
        if data:
            learner.__init__(name) # force init
            return learner(data, y)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='MV regression'):
        self.name = name

    def __call__(self, data, y, x=None, weight=None):
        
        if x == None:
            x = [v.name for v in data.domain.attributes if v.name not in y]

        dataX = data.select(x)
        dataY = data.select(y)
                
        # transformation to numpy arrays
        X = dataX.toNumpy()[0]
        Y = dataY.toNumpy()[0]
        
        # data dimensions
        n, mx = numpy.shape(X)
        my = numpy.shape(Y)[1]
        
        # Z-scores of original matrices
        YMean = numpy.mean(Y, axis=0)
        YStd = numpy.std(Y, axis=0)
        self.meanY = YMean
        self.stdY = YStd
        XMean = numpy.mean(X, axis=0)
        XStd = numpy.std(X, axis=0)
        self.meanX = XMean
        self.stdX = XStd

        X = (X - self.meanX) / self.stdX
        Y = (Y - self.meanY) / self.stdY

        self.B = dot(dot(inv(dot(X.T,X)),X.T),Y)
        self.E = dot((Y-dot(X,self.B)).T, (Y-dot(X,self.B)))
        self.x, self.y = x, y
        
        return MVRegression(model = self)
    
class MVRegression:
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, example):
       example = (numpy.array(example.native()) - self.model.meanX) / self.model.stdX
       yhat = dot(example, self.model.B) * self.model.stdY + self.model.meanY        
       return yhat
    

########################################################################
# Partial Least-Squares Regression (PLS)

# Function is used in PLSRegression
def standardize(matrix):
    "standardizes matrix and gives coloumn mean and standard deviation"
    mean = numpy.mean(matrix, axis = 0)
    one = numpy.ones(numpy.shape(matrix))
    std = numpy.std(matrix, axis = 0)
    return (matrix - numpy.multiply(mean, one))/std, mean, std

# Function is used in PLSRegression
def normalize(vector):
    return vector / numpy.linalg.norm(vector)

class PLSRegressionLearner(object):
    """PLSRegressionLearner(data, y, x=None, nc=None)"""
    def __new__(self, data=None, y=None, x=None, nc=2, name='PLS regression', **kwds):
        learner = object.__new__(self, **kwds)
        if data:
            learner.__init__(name, nc=nc) # force init
            return learner(data, y)
        else:
            return learner  # invokes the __init__

    def __init__(self, name='PLS regression', nc = 2):
        self.name = name
        self.nc = nc
        
    def __call__(self, data, y, x=None, nc=None, weight=None,
                 imputer=None, continuizer=None):
        
        if nc == None:
            nc = self.nc
        # missing values handling (impute missing)
        if imputer == None:           
            imputer = orange.ImputerConstructor_model()
            imputer.learnerContinuous = orange.MajorityLearner()
            imputer.learnerDiscrete = orange.BayesLearner()
            imputer = imputer(data)
        data = imputer(data)
 
        # continuization (replaces discrete with continuous attributes)
        # DOES NOT WORK! It needs to be fixed.
        if continuizer == None:
            continuizer = orange.DomainContinuizer()
            continuizer.multinomialTreatment = continuizer.FrequentIsBase
            continuizer.zeroBased = True
            domain0 = continuizer(data)
        data = data.translate(domain0)
        # print data.domain

        if x == None:
            x = [v.name for v in data.domain.attributes if v.name not in y]
        if nc == None:
            nc = len(x)

        X, Y = data.select(x).toNumpy()[0], data.select(y).toNumpy()[0]               
        E, self.meanX, self.stdX = standardize(X)
        F, self.meanY, self.stdY = standardize(Y) # standardized matrices
        # data dimensions
        n, mx = numpy.shape(X)
        my = numpy.shape(Y)[1] 
                
        P, C = numpy.empty((mx,nc)), numpy.empty((my,nc))
        T, U = numpy.empty((n,nc)), numpy.empty((n,nc))    
        B, W = numpy.zeros((nc,nc)), numpy.empty((mx,nc))          
        
        # main algorithm
        for i in range(nc):
            u = numpy.random.random_sample((n,1))
            w = normalize(dot(E.T,u))
            t = normalize(dot(E,w))
            dif = t
            # iterations for loading vector t
            c = normalize(dot(F.T,t))
            while numpy.linalg.norm(dif) > 10e-16:
                c = normalize(dot(F.T,t))
                u = dot(F,c)
                w = normalize(dot(E.T,u))
                t0 = normalize(dot(E,w))
                dif = t - t0
                t = t0
            T[:,i], U[:,i], C[:,i], W[:,i] = t.T, u.T, c.T, w.T
            b = dot(t.T,u)[0][0]
            B[i][i] = b
            p = dot(E.T,t)
            P[:,i] = p.T
            E, F = E - dot(t,p.T), F - b * dot(t,c.T)      
        BPls = dot(dot(numpy.linalg.pinv(P.T),B),C.T)
        
        # esimated Y
        YE = dot(dot(T,B),C.T)* self.stdY + self.meanY
        YE = dot(standardize(X)[0], BPls)* self.stdY + self.meanY  
        Y = Y * self.stdY + self.meanY
        self.W, self.T, self.P, self.U = W, T, P, U
        self.B, self.BPls, self.C = B, BPls, C
        self.x, self.y = x, y
                        
        return PLSRegression(model = self)
        

class PLSRegression:
    def __init__(self, **kwds):
        self.__dict__ = kwds

    def __call__(self, example):
       example = (numpy.array(example.native()) - self.model.meanX) / self.model.stdX
       yhat = dot(example, self.model.BPls) * self.model.stdY + self.model.meanY      
       return yhat
