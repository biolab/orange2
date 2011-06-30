from Orange import data
from Orange.classification import svm

table = data.Table("brown-selected")
print table.domain

rfe = svm.RFE()
newdata = rfe(table, 10)
print newdata.domain