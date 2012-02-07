from Orange import data
from Orange import evaluation

from Orange.classification.svm import SVMLearner, kernels
from Orange.distance import Euclidean
from Orange.distance import Hamming

iris = data.Table("iris.tab")
l1 = SVMLearner()
l1.kernel_func = kernels.RBFKernelWrapper(Euclidean(iris), gamma=0.5)
l1.kernel_type = SVMLearner.Custom
l1.probability = True
c1 = l1(iris)
l1.name = "SVM - RBF(Euclidean)"

l2 = SVMLearner()
l2.kernel_func = kernels.RBFKernelWrapper(Hamming(iris), gamma=0.5)
l2.kernel_type = SVMLearner.Custom
l2.probability = True
c2 = l2(iris)
l2.name = "SVM - RBF(Hamming)"

l3 = SVMLearner()
l3.kernel_func = kernels.CompositeKernelWrapper(
    kernels.RBFKernelWrapper(Euclidean(iris), gamma=0.5),
    kernels.RBFKernelWrapper(Hamming(iris), gamma=0.5), l=0.5)
l3.kernel_type = SVMLearner.Custom
l3.probability = True
c3 = l1(iris)
l3.name = "SVM - Composite"

tests = evaluation.testing.cross_validation([l1, l2, l3], iris, folds=5)
[ca1, ca2, ca3] = evaluation.scoring.CA(tests)

print l1.name, "CA:", ca1
print l2.name, "CA:", ca2
print l3.name, "CA:", ca3
