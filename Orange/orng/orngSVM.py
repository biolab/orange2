# ORANGE Support Vector Machines
# This module was written by Ales Erjavec
# and supersedes an earlier one written by Alex Jakulin (jakulin@acm.org),
# based on: Chih-Chung Chang and Chih-Jen Lin's
# LIBSVM : a library for support vector machines
#  (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz)

# Moved to Orange.classify.svm


try:
    import orngSVM_Jakulin
    BasicSVMLearner=orngSVM_Jakulin.BasicSVMLearner
    BasicSVMClassifier=orngSVM_Jakulin.BasicSVMClassifier
except:
    pass


from Orange.classification.svm import *
from Orange.classification.svm.kernels import *
