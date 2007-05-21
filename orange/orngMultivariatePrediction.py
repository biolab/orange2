import orange
import orngRegression

'''
def MCLearner(baseLearner, data = None, Ncomp = None, listY = None, listX = None, weightID=0, **kwds):
    l = apply(MCLearnerClass, (), kwds)
    if data:
        if baseLearner == 'PlsLearner':
            l = l(data, Ncomp, listY, listX)
        if baseLearner == 'SvmLearner':
            l = l(data, listX, listY)
    return l
'''

class MCLearnerClass:

    def __call__(self, baseLearner, data, X, Y, PLSNcomp = 3):

        if baseLearner == 'PlsLearner':
            lr = orngRegression.PLSRegressionLearner(data, PLSNcomp, Y, X)
            return lr
        if baseLearner == 'SvmLearner':
            models = []
            for i in Y:
                attr = list(X)
                attr.append(i)
                newDomain = orange.Domain([data.domain[a] for a in attr])
                newData = orange.ExampleTable(newDomain, data)
                lr = orange.SVMLearner()
                lr.svm_type=orange.SVMLearner.NU_SVR
                model = lr(newData)
                models.append(model)
            return models
                
            
d = orange.ExampleTable('C://Delo//Python//Distance Learning//04-curatedF05.tab')
ind = d.domain.index('smiles') 
nd = orange.Domain(d.domain[0:ind-1] + d.domain[ind+1:], 0)
data = orange.ExampleTable(nd, d)

lr = MCLearnerClass()('SvmLearner', data, ['growthC', 'growthE', 'dev', 'sporesC'], ['C','C-C','C-C-C'])
