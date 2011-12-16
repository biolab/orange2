from Orange import data 
from Orange.classification import svm

table = data.Table("brown-selected")
classifier = svm.SVMLearner(table, 
                            kernel_type=svm.kernels.Linear, 
                            normalization=False)

weights = svm.get_linear_svm_weights(classifier)
print weights

import pylab as plt
plt.hist(weights.values())
 