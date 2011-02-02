from Orange import core
from Orange.classification import svm

data = core.ExampleTable("brown-selected")
print data.domain

rfe = svm.RFE()
newdata = rfe(data, 10)
print newdata.domain