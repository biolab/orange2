from Orange import data 
from Orange.classification import svm

brown = data.Table("brown-selected")
classifier = svm.SVMLearner(brown, 
                            kernel_type=svm.kernels.Linear, 
                            normalization=False)

weights = svm.get_linear_svm_weights(classifier)
print weights

import pylab as plt
plt.hist(weights.values())
 