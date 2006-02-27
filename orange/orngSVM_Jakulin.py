# ORANGE Support Vector Machines
#    by Alex Jakulin (jakulin@acm.org)
#
#       based on:
#           Chih-Chung Chang and Chih-Jen Lin
#           LIBSVM : a library for support vector machines.
#           http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.ps.gz
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# CVS Status: $Id$ 
#
# Version 1.8 (18/11/2003)
#   - added error checking, updated to libsvm 2.5
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
#

import orange
import orngCRS
import orng2Array
import math

class BasicSVMLearner(orange.Learner):
  def __init__(self):
      self._name = "SVM Learner Wrap"
      # "SVM type (C_SVC=0, NU_SVC, ONE_CLASS, EPS_SVR, NU_SVR=4)
      # SV-classifier : "ordinary" SVM
      # nu-SV-classifier : nu controls the complexity of the model
      # ONE_CLASS: only one class -- is something in or out
      # epsilon-SVR: epsilon-regression
      # epsilon-SVR: nu-regression
      self.type = -1 # -1: classical, -2: NU, -3: OC

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

      # probability (1 = on, 0 = off)
      self.probability = 0

      # class weights
      self.classweights = []

      self.translation_mode_d = 1
      self.translation_mode_c = 1
      self.for_nomogram = 0

      self.normalize = 0      
      
  def getmodel(self,data,fulldata):
      # make sure that regression is used for continuous classes, and classification
      # for discrete class
      assert(data.domain.classVar.varType == 1 or data.domain.classVar.varType == 2)
      typ = self.type
      if typ == -1: # Classical
        if data.domain.classVar.varType == 2: # continuous class
          typ = 3 # regression
        else: # discrete class
          typ = 0 # classification
      elif typ == -2: # Nu
        if data.domain.classVar.varType == 2: # continuous class
          typ = 4 # regression
        else: # discrete class
          typ = 1 # classification
      elif typ == -3: # OC
        typ = 2 # one-class, class is ignored.

      # do error checking
      if type(self.degree) == type(1):
          self.degree = float(self.degree)
      if type(self.cache_size) == type(1):
          self.cache_size = float(self.cache_size)
      assert(type(self.degree) == type(1.0))
      assert(type(self.gamma) == type(1.0))
      assert(type(self.coef0) == type(1.0))
      assert(type(self.nu) == type(1.0))
      assert(type(self.cache_size) == type(1.0))
      assert(type(self.C) == type(1.0))
      assert(type(self.eps) == type(1.0))
      assert(type(self.p) == type(1.0))
      assert(typ in [0,1,2,3,4])
      assert(self.kernel in [0,1,2,3])
      assert(self.cache_size > 0)
      assert(self.eps > 0)
      assert(self.nu <= 1.0 and self.nu >= 0.0)
      assert(self.p >= 0.0)
      assert(self.shrinking in [0,1])
      assert(self.probability in [0,1]) 
      if type == 1:
        counts = [0]*len(data.domain.classVar.values)
        for x in data:
          counts[int(x.getclass())] += 1
        for i in range(1,len(counts)):
          for j in range(i):
            if self.nu*(counts[i]+counts[j]) > 2*min(counts[i],counts[j]):
              raise "Infeasible nu value."

      puredata = orange.Filter_hasClassValue(data)
      translate = orng2Array.DomainTranslation(self.translation_mode_d,self.translation_mode_c)
      if fulldata != 0:
          purefulldata = orange.Filter_hasClassValue(fulldata)
          translate.analyse(purefulldata)
      else:
          translate.analyse(puredata)
      translate.prepareSVM(not self.for_nomogram)
      mdata = translate.transform(puredata)

      if len(self.classweights)==0:
          model = orngCRS.SVMLearn(mdata, typ, self.kernel, self.degree, self.gamma, self.coef0, self.nu, self.cache_size, self.C, self.eps, self.p, self.shrinking, self.probability, 0, [], [])
      else:
          assert(len(puredata.domain.classVar.values)==len(self.classweights))
          cvals = [data.domain.classVar(i) for i in data.domain.classVar.values]
          labels = translate.transformClass(cvals)
          model = orngCRS.SVMLearn(mdata, typ, self.kernel, self.degree, self.gamma, self.coef0, self.nu, self.cache_size, self.C, self.eps, self.p, self.shrinking, self.probability, len(self.classweights), self.classweights, labels)
      return (model, translate)

  def __call__(self, data, weights = 0,fulldata=0):
      # note that weights are ignored
      (model, translate) = self.getmodel(data,fulldata)
      return BasicSVMClassifier(model,translate,normalize=(self.normalize or self.for_nomogram))

class BasicSVMClassifier(orange.Classifier):
  def __init__(self, model, translate, normalize):
      self._name = "SVM Classifier Wrap"
      self.model = model
      self.cmodel = orngCRS.SVMClassifier(model)
      self.translate = translate
      self.normalize = normalize
      if model["svm_type"] in [0,1]:
          self.classifier = 1
          if model.has_key("ProbA") and model.has_key("ProbB"):
              self.probabilistic = 1
          else:
              self.probabilistic = 0
          self.classLUT = [self.translate.getClass(q) for q in model["label"]]
          self.iclassLUT = [int(q) for q in self.classLUT]
      else:
          self.probabilistic = 0
          self.classifier = 0
          
      if normalize and model['kernel_type'] == 0 and model["svm_type"] == 0 and model["nr_class"] == 2:
          beta = model["rho"][0]
          svs = model["SV"]
          ll = -1
          for i in xrange(model["total_sv"]):
              ll = max(ll,svs[i][-1][0])
          xcoeffs = [0.0]*(ll)
          for i in xrange(model["total_sv"]):
              csv = svs[i]
              coef = csv[0][0]
              for (j,v) in csv[1:]:
                  xcoeffs[j-1] += coef*v
          sum = 0.0
          for x in xcoeffs:
              sum += x*x
          self.coefficient = 1.0/math.sqrt(sum)
          self.xcoeffs = [x*self.coefficient for x in xcoeffs]
          self.beta = beta*self.coefficient
      else:
          self.coefficient = 1.0
          
  def getmargin(self, example):
      # classification with margins
      assert(self.model['nr_class'] <= 2) # this should work only with 2-class problems
      if self.model['nr_class'] == 2:
        td = self.translate.extransform(example)
        margin = orngCRS.SVMClassifyM(self.cmodel,td)
        if self.normalize:
            return margin[0]*self.coefficient
        else:
            return margin[0]
      else:
        # it can happen that there is a single class
        return 0.0

  def __call__(self, example, format = orange.GetValue):
      # classification
      td = self.translate.extransform(example)
      x = orngCRS.SVMClassify(self.cmodel,td)
      v = self.translate.getClass(x)
      if self.probabilistic:
          px = orngCRS.SVMClassifyP(self.cmodel,td)
          p = [0.0]*len(self.translate.cv.attr.values)
          for i in xrange(len(self.iclassLUT)):
              p[self.iclassLUT[i]] = px[i]
      elif self.model['svm_type']==0 or self.model['svm_type']==1:
          p = [0.0]*len(self.translate.cv.attr.values)
          p[int(v)] = 1.0

      if format == orange.GetValue or self.model['svm_type']==3 or self.model['svm_type']==2:
          # do not return and PD when we're dealing with regression, or one-class
          return v
      if format == orange.GetBoth:
          return (v,p)
      if format == orange.GetProbabilities:
          return p

  def __del__(self):
    orngCRS.svm_destroy_model(self.cmodel)
