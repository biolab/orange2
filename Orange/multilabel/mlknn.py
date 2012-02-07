""" 
.. index:: ML-kNN Learner

***************************************
ML-kNN Learner
***************************************

ML-kNN Classification is a kind of adaptation method for multi-label classification.
It is an adaptation of the kNN lazy learning algorithm for multi-label data.
In essence, ML-kNN uses the kNN algorithm independently for each label :math:`l`.
It finds the k nearest examples to the test instance and considers those that are
labeled at least with :math:`l` as positive and the rest as negative.
Actually this method follows the paradigm of Binary Relevance (BR). What mainly
differentiates this method from BR is the use of prior probabilities. ML-kNN has also
the capability of producing a ranking of the labels as an output.
For more information, see Zhang, M. and Zhou, Z. 2007. `ML-KNN: A lazy learning
approach to multi-label learning <http://dx.doi.org/10.1016/j.patcog.2006.12.019>`_. 
Pattern Recogn. 40, 7 (Jul. 2007), 2038-2048.  

.. index:: ML-kNN Learner
.. autoclass:: Orange.multilabel.MLkNNLearner
   :members:
   :show-inheritance:

   :param instances: a table of instances.
   :type instances: :class:`Orange.data.Table`
 

.. index:: ML-kNN Classifier
.. autoclass:: Orange.multilabel.MLkNNClassifier
   :members:
   :show-inheritance:

Examples
========

The following example demonstrates a straightforward invocation of
this algorithm (:download:`mlc-classify.py <code/mlc-classify.py>`):

.. literalinclude:: code/mlc-classify.py
   :lines: 6, 11-13

"""
import random
import Orange
import multiknn as _multiknn

from lp import transform_to_powerset

class MLkNNLearner(_multiknn.MultikNNLearner):
    """
    Class implementing the ML-kNN (Multi-Label k Nearest Neighbours) algorithm. The class is based on the 
    pseudo-code made available by the authors.
    
    The pseudo code of ML-kNN:
    
        :math:`[\\vec y_t,\\vec r_t] = ML-kNN(T,K,t,s)`
    
        :math:`\%Computing \\quad the \\quad prior \\quad probabilities \\quad P(H_b^l)`
    
        :math:`(1) for \\quad l \\in y \\quad do`

        :math:`(2) \\quad  P(H_1^l) = (s+ \\sum_{i=1}^m \\vec y_{x_i}(l))/(s * 2+m); P(H_0^l)=1-P(H_1^l)`

        :math:`\%Computing \\quad the \\quad posterior \\quad probabilities P(E_j^l|H_b^l)`
    
        :math:`(3) Identify \\quad N(x_i), i \\in {1,2,...,m}`
        
        :math:`(4) for \\quad l \\in y \\quad do`
        
        :math:`(5) \\quad for \\quad j \\in{0,1,...,K} \\quad do`
        
        :math:`(6) \\quad \\quad c[j]=0; c'[j]=0`
        
        :math:`(7) \\quad for \\quad i \\in{1,...,m} \\quad do`
        
        :math:`(8) \\quad \\quad \\delta = \\vec C_{x_i}(l)=\\sum_{a \\in N(x_i)} \\vec y_a(l)`
        
        :math:`(9) \\quad \\quad if (\\vec y_{x_i}(l)==1) \\quad then \\quad c[\\delta]=c[\\delta]+1`
        
        :math:`(10)\\quad \\quad \\quad \\quad else \\quad c'[\\delta]=c'[\\delta]+1`
        
        :math:`(11)\\quad for \\quad j \\in{0,1,...,K} \\quad do`
        
        :math:`(12)\\quad \\quad P(E_j^l|H_1^l)=(s+c[j])/(s * (K+1)+ \\sum_{p=0}^k c[p])`
        
        :math:`(13)\\quad \\quad P(E_j^l|H_0^l)=(s+c'[j])/(s *(K+1)+ \\sum_{p=0}^k c'[p])`
        
        :math:`\%Computing \\quad \\vec y_t \\quad and \\quad \\vec r_t` 
        
        :math:`(14) Identify \\quad N(t)`
        
        :math:`(15) for \\quad l \\in y \\quad do`
        
        :math:`(16)\\quad \\vec C_t(l) = \\sum_{a \\in N(t)} \\vec y_a(L)`
        
        :math:`(17)\\quad \\vec y_t(l) = argmax_{b \\in {0,1}}P(H_b^l)P(E_{\\vec C_t(l)}^l|H_b^l)`
        
        :math:`(18)\\quad \\vec r_t(l)=P(H_1^l|E_{\\vec C_t(l)}^l)=P(H_1^l)P(E_{\\vec C_t(l)}|H_1^l)/P(E_{\\vec C_t(l)}^l)=P(H_1^l)P(E_{\\vec C_t(l)}|H_1^l)/(\\sum_{b \\in {0,1}}P(H_b^l)P(E_{\\vec C_t(l)}^l|H_b^l))`

     
    .. attribute:: k
    
        Number of neighbors. The default value is 1 
    
    .. attribute:: smooth
    
        Smoothing parameter controlling the strength of uniform prior (Default value is set to 1 which yields the Laplace smoothing).
    
    .. attribute:: knn
        
        :class:`Orange.classification.knn.FindNearest` for nearest neighbor search
    
    """
    def __new__(cls, instances = None, k=1, smooth = 1.0, weight_id = 0, **argkw):
        """
        Constructor of MLkNNLearner
        
        :param instances: a table of instances.
        :type instances: :class:`Orange.data.Table`
        
        :param k: number of nearest neighbors used in classification
        :type k: int
        
        :param smooth: Smoothing parameter controlling the strength of uniform prior 
        (Default value is set to 1 which yields the Laplace smoothing).
        :type smooth: Float
        
        :rtype: :class:`MLkNNLearner`
        """
        
        self = _multiknn.MultikNNLearner.__new__(cls, k, **argkw)
        self.smooth = smooth
        
        if instances:
            self.__init__(**argkw)
            return self.__call__(instances,weight_id)
        else:
            return self

    def __call__(self, instances, weight_id = 0, **kwds):
        if not Orange.multilabel.is_multilabel(instances):
            raise TypeError("The given data set is not a multi-label data set.")
        
        self.__dict__.update(kwds)
        self._build_knn(instances)

        #Computing the prior probabilities P(H_b^l)
        prior_prob = self.compute_prior(instances)
        
        #Computing the posterior probabilities P(E_j^l|H_b^l)
        cond_prob = list(self.compute_cond(instances))
        
        return MLkNNClassifier(instances = instances,
                               prior_prob = prior_prob, 
                               cond_prob = cond_prob,
                               knn = self.knn,
                               k = self.k)

    def compute_prior(self, instances):
        """ Compute prior probability for each label of the training set. """
        prior_prob = []
        for lvar in instances.domain.class_vars:
            freq = sum(inst[lvar].value == '1' for inst in instances)
            prior_prob.append( float(self.smooth + freq) / (self.smooth * 2 + len(instances)) )
        return prior_prob
            
    def compute_cond(self, instances):
        """ Compute posterior probabilities for each label of the training set. """
        k = self.k
        
        def _remove_identical(table, inst):
            try:
                i = [inst1.get_classes() == inst.get_classes() for inst1 in table].index(1)
            except:
                i = -1
            del table[i]
            return table
            
            
        neighbor_lists = [_remove_identical(self.knn(inst, k+1), inst) for inst in instances]
        p1 = [[0]*(k+1) for lvar in instances.domain.class_vars]
        p0 = [[0]*(k+1) for lvar in instances.domain.class_vars]

        for li, lvar in enumerate(instances.domain.class_vars):
            c  = [0] * (k + 1)
            cn = [0] * (k + 1)
            
            for inst, neighbors in zip(instances, neighbor_lists):
                delta = sum(n[lvar].value=='1' for n in neighbors)
                
                (c if inst[lvar].value == '1' else cn)[delta] += 1
                
            for j in range(k+1):
                p1[li][j] = float(self.smooth + c[j]) / (self.smooth * (k+1) + sum(c))
                p0[li][j] = float(self.smooth + cn[j]) / (self.smooth * (k+1) + sum(cn))
        
        return p0, p1
 
class MLkNNClassifier(_multiknn.MultikNNClassifier):      
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        """
        :rtype: a list of :class:`Orange.data.Value`, a list of :class:`Orange.statistics.distribution.Distribution`, or a tuple with both
        """
        neighbors = self.knn(instance, self.k)
        
        labels = []
        dists = []
        
        for li, lvar in enumerate(self.instances.domain.class_vars):
            delta = sum(n[lvar].value=='1' for n in neighbors)
    
            p1 = self.prior_prob[li] * self.cond_prob[1][li][delta]
            p0 = (1-self.prior_prob[li]) * self.cond_prob[0][li][delta]
            y = (p1 >= p0)
            labels.append(Orange.data.Value(lvar, str(int(y))))
            
            r = p1 / (p0+p1)
            dists.append( Orange.statistics.distribution.Discrete([1-r, r]) )
       
        for d, lvar in zip(dists, self.instances.domain.class_vars):
            d.variable = lvar
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return dists
        return labels, dists
        
#########################################################################################
# Test the code, run from DOS prompt
# assume the data file is in proper directory

if __name__ == "__main__":
    data = Orange.data.Table("emotions.tab")

    classifier = Orange.multilabel.MLkNNLearner(data,5,1.0)
    for i in range(10):
        c,p = classifier(data[i],Orange.classification.Classifier.GetBoth)
        print c,p