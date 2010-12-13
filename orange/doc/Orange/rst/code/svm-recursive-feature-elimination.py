import orange
import Orange.classify.svm as svm

data = orange.ExampleTable("brown-selected")
print data.domain

rfe = svm.RFE()
newdata = rfe(data, 10)
print newdata.domain