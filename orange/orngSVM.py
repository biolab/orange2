# ORANGE Support Vector Machines
#    by Alex Jakulin (jakulin@acm.org)
#
#       based on:
#           Chih-Chung Chang and Chih-Jen Lin
#           LIBSVM : a library for support vector machines.
#           http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Version 1.7 (11/08/2002)
#   - Assertion error was resulting because Orange incorrectly compared the
#     attribute values as returned from .getclass() and the values array in
#     the attribute definition. Cast both values to integer before comparison.
#   - Support for setting the preferred ordinal attribute transformation.
#
# Version 1.6 (31/10/2001)
#
# To Do:
#   - class-weighting SVM (each class c of c_1..c_k is weighted by k/p(c))
#
#   - make sure everything works when given an empty example table
#
#   - Consider supporting TRANSDUCTION. Currently all the missing-class examples
#     are filtered out.
#

import orange
import orngCRS
import orng2Array

class BasicSVMLearner:
  def __init__(self):
      self.name = "SVM Learner Wrap"
      # "SVM type (C_SVC=0, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR=4)
      # SV-classifier : "ordinary" SVM
      # nu-SV-classifier : nu controls the complexity of the model
      # ONE_CLASS: only one class -- is something in or out
      # epsilon-SVR: epsilon-regression
      # epsilon-SVR: nu-regression
      self.type = 0

      # kernel type: (LINEAR=0, POLY, RBF, SIGMOID=3)
      # linear: x[i].x[j]
      # poly: pow(gamma*x[i].x[j]+coef0, degree)
      # rbf: exp(-gamma*(x[i]^2+x[j]^2-2*x[i].x[j]))
      # sigm: tanh(gamma*x[i].x[j]+coef0)
      self.kernel = 2

      # poly degree when POLY
      self.degree = 3
      
      # poly/rbf/sigm parameter
      # if 0.0, it is assigned the default value of 1.0/#attributes
      self.gamma = 0.0

      # poly/sigm      
      self.coef0 = 0.0

      # complexity control with NU_SVC, NU_SVR in ONE_CLASS,
      # the bigger, the less complex the model 
      self.nu = 0.5

      # cache size in MB
      self.cache_size = 40

      # for C_SVR
      self.p = 0.5

      # for SVC, SVR and NU_SVR
      # greater the cost of misclassification, greater the likelihood of overfitting
      self.C = 1.0

      # tolerance
      self.eps = 1e-3

      # shrinking heuristic (1=on, 0=off)
      self.shrinking = 1

      # class weights
      self.classweights = []

      self.translation_mode = 1      
      
  def getmodel(self,data):
      # make sure that regression is used for continuous classes, and classification
      # for discrete class
      assert(data.domain.classVar.varType == 1 or data.domain.classVar.varType == 2)
      type = self.type
      if data.domain.classVar.varType == 2: # continuous class
        type = 3 # regression
      else: # discrete class
        if type == 3: # if using regression
          type = 0 # switch to classification
      puredata = orange.Filter_hasClassValue(data)
      translate = orng2Array.DomainTranslation(self.translation_mode)
      translate.analyse(puredata)
      translate.prepareSVM()
      mdata = translate.transform(puredata)
      if len(self.classweights)==0:
          #import pickle
          #r = (mdata, type, self.kernel, self.degree, self.gamma, self.coef0, self.nu, self.cache_size, self.C, self.eps, self.p, self.shrinking)
          #pickle.dump(r,open('c:/temp/huh.pik','w'))
          model = orngCRS.SVMLearn(mdata, type, self.kernel, self.degree, self.gamma, self.coef0, self.nu, self.cache_size, self.C, self.eps, self.p, self.shrinking)
          #print model
      else:
          assert(len(puredata.domain.classVar.values)==len(self.weights))
          cvals = [data.domain.classVar(i) for i in data.domain.classVar.values]
          labels = translate.transformClass(cvals)
          model = orngCRS.SVMLearn(mdata, type, self.kernel, self.degree, self.gamma, self.coef0, self.nu, self.cache_size, self.C, self.eps, self.p, self.shrinking,len(self.classweights), self.classweights, labels)
      return (model, translate)

  def __call__(self, data, weights = 0):
      # note that weights are ignored
      (model, translate) = self.getmodel(data)
      return BasicSVMClassifier(model,translate)


class BasicSVMClassifier:
  def __init__(self, model, translate):
      self.name = "SVM Classifier Wrap"
      self.model = model
      self.cmodel = orngCRS.SVMClassifier(model)
      self.translate = translate

  def getmargin(self, example):
      # classification with margins
      assert(self.model['nr_class'] <= 2) # this should work only with 2-class problems
      if self.model['nr_class'] == 2:
        td = self.translate.extransform(example)
        margin = orngCRS.SVMClassifyM(self.cmodel,td)
        return margin
      else:
        # it can happen that there is a single class
        return 0.0

  def __call__(self, example, format = orange.GetValue):
      # classification
      td = self.translate.extransform(example)
      x = orngCRS.SVMClassify(self.cmodel,td)
      v = self.translate.getClass(x)
      if format == orange.GetValue or self.model['svm_type']==3 or self.model['svm_type']==2:
          # do not return and PD when we're dealing with regression, or one-class
          return v
      p = [0.0]*len(self.translate.cv.attr.values)
      for i in range(len(self.translate.cv.attr.values)):
          if int(v) == i:
              p[i] = 1.0
              break
      if format == orange.GetBoth:
          return (v,p)
      if format == orange.GetProbabilities:
          return p
