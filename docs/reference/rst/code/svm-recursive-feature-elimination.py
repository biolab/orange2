from Orange import data
from Orange.classification import svm

brown = data.Table("brown-selected")
print brown.domain

rfe = svm.RFE()
newdata = rfe(brown, 10)
print newdata.domain