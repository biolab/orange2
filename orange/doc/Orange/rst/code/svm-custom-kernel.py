from Orange import core
from Orange.classification import svm

data=core.ExampleTable("iris.tab")
l1=svm.SVMLearner()
l1.kernelFunc=svm.kernels.RBFKernelWrapper(core.ExamplesDistanceConstructor_Euclidean(data), gamma=0.5)
l1.kernel_type=svm.SVMLearner.Custom
l1.probability=True
c1=l1(data)
l1.name="SVM - RBF(Euclidean)"

l2=svm.SVMLearner()
l2.kernelFunc=svm.kernels.RBFKernelWrapper(
    core.ExamplesDistanceConstructor_Hamming(data), gamma=0.5)
l2.kernel_type=svm.SVMLearner.Custom
l2.probability=True
c2=l2(data)
l2.name="SVM - RBF(Hamming)"

l3=svm.SVMLearner()
l3.kernelFunc = svm.kernels.CompositeKernelWrapper(
    svm.kernels.RBFKernelWrapper(
    core.ExamplesDistanceConstructor_Euclidean(data), gamma=0.5),
    svm.kernels.RBFKernelWrapper(
    core.ExamplesDistanceConstructor_Hamming(data), gamma=0.5), l=0.5)
l3.kernel_type=svm.SVMLearner.Custom
l3.probability=True
c3=l1(data)
l3.name="SVM - Composite"


import orngTest, orngStat
tests=orngTest.crossValidation([l1, l2, l3], data, folds=5)
[ca1, ca2, ca3]=orngStat.CA(tests)
print l1.name, "CA:", ca1
print l2.name, "CA:", ca2
print l3.name, "CA:", ca3