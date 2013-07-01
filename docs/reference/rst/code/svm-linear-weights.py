from Orange import data
from Orange.classification import svm

brown = data.Table("brown-selected")
classifier = svm.SVMLearner(brown,
                            kernel_type=svm.kernels.Linear,
                            normalization=False,
                            eps=1e-9)

weights = svm.get_linear_svm_weights(classifier)
print sorted("%.4f" % w for w in weights.values())

import pylab as plt
plt.hist(weights.values())
