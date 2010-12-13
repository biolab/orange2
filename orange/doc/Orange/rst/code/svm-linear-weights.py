import orange
import Orange.classify.svm as svm

data = orange.ExampleTable("brown-selected")
classifier = svm.SVMLearner(data, kernel_type=svm.Linear, normalization=False)

weights = svm.getLinearSVMWeights(classifier)
print weights

import pylab as plt
plt.hist(weights.values())
