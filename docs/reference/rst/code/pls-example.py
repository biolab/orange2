# Description: Partial least squares regression
# Category:    regression
# Uses:        multitarget-synthetic
# Referenced:  Orange.regression.pls
# Classes:     Orange.regression.pls.PLSRegressionLearner, Orange.regression.pls.PLSRegression

import Orange

data = Orange.data.Table("multitarget-synthetic.tab")
print "Input variables:    ", data.domain.features
print "Response variables: ", data.domain.class_vars
    
learner = Orange.multitarget.pls.PLSRegressionLearner()
classifier = learner(data)

print "Prediction for the first 2 data instances: \n" 
for d in data[:2]:
    print "Actual    ", d.get_classes()
    print "Predicted ", classifier(d)
    print 

print 'Regression coefficients:\n', classifier    
