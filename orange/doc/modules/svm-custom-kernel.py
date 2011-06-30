import orange, orngSVM
data=orange.ExampleTable("iris.tab")
l1=orngSVM.SVMLearner()
l1.kernelFunc=orngSVM.RBFKernelWrapper(orange.ExamplesDistanceConstructor_Euclidean(data), gamma=0.5)
l1.kernel_type=orange.SVMLearner.Custom
l1.probability=True
c1=l1(data)
l1.name="SVM - RBF(Euclidean)"

l2=orngSVM.SVMLearner()
l2.kernelFunc=orngSVM.RBFKernelWrapper(orange.ExamplesDistanceConstructor_Hamming(data), gamma=0.5)
l2.kernel_type=orange.SVMLearner.Custom
l2.probability=True
c2=l2(data)
l2.name="SVM - RBF(Hamming)"

l3=orngSVM.SVMLearner()
l3.kernelFunc=orngSVM.CompositeKernelWrapper(orngSVM.RBFKernelWrapper(orange.ExamplesDistanceConstructor_Euclidean(data), gamma=0.5),orngSVM.RBFKernelWrapper(orange.ExamplesDistanceConstructor_Hamming(data), gamma=0.5), l=0.5)
l3.kernel_type=orange.SVMLearner.Custom
l3.probability=True
c3=l1(data)
l3.name="SVM - Composite"


import orngTest, orngStat
tests=orngTest.crossValidation([l1, l2, l3], data, folds=5)
[ca1, ca2, ca3]=orngStat.CA(tests)
print l1.name, "CA:", ca1
print l2.name, "CA:", ca2
print l3.name, "CA:", ca3