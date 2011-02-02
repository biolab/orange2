from Orange import core 
from Orange.classification import svm

data = core.ExampleTable("brown-selected")
classifier = svm.SVMLearner(data, kernel_type=svm.kernels.Linear, normalization=False)

weights = svm.getLinearSVMWeights(classifier)
print weights

import pylab as plt
plt.hist(weights.values())
