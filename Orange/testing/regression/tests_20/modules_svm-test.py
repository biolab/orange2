import orange, orngSVM
data = orange.ExampleTable("iris.tab")
lin = orngSVM.SVMLearner(kernel_type=orngSVM.SVMLearner.Linear, name="SVM - Linear")
poly = orngSVM.SVMLearner(kernel_type=orngSVM.SVMLearner.Polynomial, name="SVM - Poly")
rbf = orngSVM.SVMLearner(kernel_type=orngSVM.SVMLearner.RBF, name="SVM - RBF")

learners = [lin, poly, rbf]
import orngTest, orngStat
res = orngTest.crossValidation(learners, data)
print "%15s%8s%8s" % ("Name", "CA", "AUC")
for l, ca, auc in zip(learners, orngStat.CA(res), orngStat.AUC(res)):
  print "%-15s   %.3f   %.3f" % (l.name, ca, auc)
