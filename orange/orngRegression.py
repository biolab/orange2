import orange
import numpy
from numpy.linalg import inv
from numpy import dot, sqrt
import statc
from string import join

# Function is used in PLSRegression
def standardize(matrix):
    mean = numpy.mean(matrix, axis = 0)
    one = numpy.ones(numpy.shape(matrix))
    std = numpy.std(matrix, axis = 0)
    return (matrix - numpy.multiply(mean, one))/std

# Function is used in PLSRegression
def normalize(vector):
    return vector / numpy.linalg.norm(vector)


def LinearRegression(examples=None, weightID=0, **kwds):
    l=apply(LinearRegressionLearner, (), kwds)
    if examples:
        l=l(examples)
    return l

class LinearRegressionLearner:
    
    def __init__(self, beta0 = True):
        self.beta0 = beta0
    
    def __call__(self, data):
       #   missing values
       #
       #   imputation
       #   for dicrete atributes uses BayesLearner
       #   and MajorityLearner for continuous
       imputer = orange.ImputerConstructor_model()
       imputer.learnerContinuous = orange.MajorityLearner()
       imputer.learnerDiscrete = orange.BayesLearner()
       imputer = imputer(data)
       data = imputer(data)

       #   binarization
       continuizer = orange.DomainContinuizer()
       continuizer.multinomialTreatment = continuizer.FrequentIsBase
       continuizer.zeroBased = True
       domain0 = continuizer(data)
       self.data = data.translate(domain0)

       #   convertion to numpy
       A, y, w = self.data.toNumpy()        # weights ??
       n, m = numpy.shape(A)
    
       if self.beta0 == True:
            X = numpy.insert(A,0,1,axis=1) # adds a column of ones
            
       else:
            X = A
            
       beta, resid, rank, s = numpy.linalg.lstsq(X,y)  #ne uposteva (ne)polnosti ranga--- mozna singularnost
       
       yEstimated = dot(X,beta)  # estimation

       #   some desriptive statistisc
       muY, sigmaY = numpy.mean(y), numpy.std(y)
       muX, covX = numpy.mean(A, axis = 0), numpy.cov(A, rowvar = 0)

       #   some statistics
       SST, SSR = numpy.sum((y - muY) ** 2), numpy.sum((yEstimated - muY) ** 2)
       SSE, RSquare = SST-SSR, SSR/SST
       R = numpy.sqrt(RSquare) # coefficient of determination
       RAdjusted = 1 - (1 - RSquare) * (n - 1) / (n - m - 1)
       F = (SSR / m) / (SST - SSR / (n - m - 1)) # F statistisc
       df = m - 1

       sigmaSquare = SSE / (n-m-1)
       #   standard error of estimated coefficients
       errCoeff = sqrt(sigmaSquare * inv(dot(X.T,X)).diagonal())

       #   t statistisc, significance,...
       t = beta / errCoeff
       Significance = [statc.betai(df*0.5,0.5,df/(df+tt*tt)) for tt in t]

       #   standardized coefficients
       if self.beta0 == True:   
            stdCoeff = (sqrt(covX.diagonal()) / sigmaY)  * beta[1:]
       else:
            stdCoeff = (sqrt(covX.diagonal()) / sigmaY)  * beta

       statistics ={'descriptives': { 'meanX' : muX, 'covX' : covX, 'meanY' : muY, 'sigmaY' : sigmaY},
                    'model' : {'estCoeff' : beta, 'stdErrorEstimation': errCoeff},
                    'model summary': {'TotalVar' : SST, 'ExplVar' : SSE, 'ResVar' : SSR, 'R' : R, 'RAdjusted' : RAdjusted,
                                      'F' : F, 't' : t, 'sig': Significance}
                    }

       return LinearRegressionClass(self.data, statistics, self.beta0)


class LinearRegressionClass:
   def __init__(self, data, statistics, beta0):
       self.data = data
       self.statistics = statistics
       self.beta = statistics['model']['estCoeff']
       self.beta0 = beta0

   def __call__(self, example):
       cexample = orange.Example(self.data.domain, example)
       cexample = numpy.array(cexample.native())
       if self.beta0 == True:
           value = self.beta[0] + dot(self.beta[1:],cexample[:-1])
       else:
           value = dot(self.beta,cexample[:-1])
        
       return value

   def prnt(self):
       """prints some basic statistics of regression model"""
       beta = self.beta
       data = self.data
       err = self.statistics['model']['stdErrorEstimation']
       t = self.statistics['model summary']['t']
       sig = self.statistics['model summary']['sig'] 
       beta0 = self.beta0

       labels = ('Variable', 'Coeff Est', 'Std Error', 't-value', 'p')
       print join(['%10s' % l for l in labels], ' ')

       fmt = "%10s " + join(["%10.3f"]*4, " ")
       if beta0 == True:
            print fmt % ('Constant', beta[0], err[0], t[0], sig[0])
            for i in range(len(data[0])-1):
                print fmt % (data.domain.attributes[i].name, beta[i+1], err[i+1], t[i+1], sig[i+1])
       else:
            for i in range(len(data[0])-1):
                print fmt % (data.domain.attributes[i].name, beta[i], err[i], t[i], sig[i])       


#
#
#       PLS Regression
#
#


def PLSRegressionLearner(data = None, Ncomp = None, listY = None, listX = None, weightID=0, **kwds):
    ''' Forms PLS regression model from data (example table)
        using Ncomp components. Names of independent
        variables are stored in list listX and names of response
        variables are stored in list listY.
    '''
    l = apply(PLSRegressionLearnerClass, (), kwds)
    if data:
        l = l(data, Ncomp, listY, listX)
    return l


class PLSRegressionLearnerClass:
    
    def __call__(self, data, Ncomp, listY, listX = None):
        
        if listX == None:
            listX = []
            for i in data.domain.variables:
                if listY.count(i.name) == 0:
                    listX.append(i)

        dataX = data.select(listX)
        dataY = data.select(listY)
        
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
        return PLSRegression(data, BPls, YMean, YStd, XMean, XStd)

class PLSRegression:
    
   def __init__(self, data, BPls, YMean, YStd, XMean, XStd):
       self.data = data
       self.BPls = BPls
       self.YMean = YMean
       self.YStd = YStd
       self.XMean = XMean
       self.XStd = XStd

   def __call__(self, example):
       example = numpy.array(example.native())
       example = (example - self.XMean) / self.XStd
       estimator = dot(example, self.BPls) * self.YStd + self.YMean        
       return estimator

    

d = orange.ExampleTable('C://Delo//Python//Distance Learning//04-curatedF05.tab')
ind = d.domain.index('smiles') 
nd = orange.Domain(d.domain[0:ind-1] + d.domain[ind+1:], 0)
data = orange.ExampleTable(nd, d)

selY = ['growthC', 'growthE', 'dev', 'sporesC']

selX = []
for i in data.domain.variables:
    if selY.count(i.name) == 0:
        selX.append(i)
        
dataX = data.select(selX)
dataY = data.select(selY)

model = PLSRegressionLearner(data, 3, selY, selX)
        
X0 = dataX.toNumpy()[0]
Y0 = dataY.toNumpy()[0]

regressor = PLSRegression(model.data, model.BPls, model.YMean, model.YStd, model.XMean, model.XStd)
est = regressor(dataX[1])
print 'Original values:', dataY[1]
print 'Estimated values:', est 
print '\n'

   

if __name__ == "__main__":
   data = orange.ExampleTable("C://ds//housing.tab")
   lr = LinearRegression(data)
   lr.prnt()
   
 
