""" 
.. index:: ML-kNN Learner

***************************************
ML-kNN Learner
***************************************

ML-kNN Classification is a kind of adaptation method for multi-label classification. 
It is an adaptation of the kNN lazy learning algorithm for multi-label data. 
In essence, ML-kNN uses the kNN algorithm independently for each label :math:'l': 
It finds the k nearest examples to the test instance and considers those that are 
labelled at least with :math:'l' as positive and the rest as negative. 
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
 
   .. method:: __new__(instances, **argkw) 
   MLkNNLearner Constructor
   
   :param instances: a table of instances, covered by the rule.
   :type instances: :class:`Orange.data.Table`

.. index:: MLkNN Classifier
.. autoclass:: Orange.multilabel.MLkNNClassifier
   :members:
   :show-inheritance:

   .. method:: __call__(self, example, result_type)
   :rtype: a list of :class:`Orange.data.Value`, 
              :class:`Orange.statistics.Distribution` or a tuple with both 
   
Examples
========

The following example demonstrates a straightforward invocation of
this algorithm (`mlc-classify.py`_, uses `multidata.tab`_):

.. literalinclude:: code/mlc-classify.py
   :lines: 1-3, 24-27

.. _mlc-classify.py: code/mlc-example.py
.. _multidata.tab: code/multidata.tab

"""
import random
import Orange
import label
import multiknn as _multiknn

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
        
        :param instances: a table of instances, covered by the rule.
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
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]

        _multiknn.MultikNNLearner.transfor_table(self,instances)

        num_labels = self.num_labels
        k = self.k
        
        #A table holding the prior probability for an instance to belong in each class
        self.prior_probabilities  = [0.] * num_labels
        
        #A table holding the prior probability for an instance not to belong in each class
        self.prior_nprobabilities = [0.] * num_labels
        
        #A table holding the probability for an instance to belong in each class given that i:0..k of its neighbors belong to that class
        self.cond_probabilities   = [ [0.] * (k + 1) ] * num_labels
        
        #A table holding the probability for an instance not to belong in each class given that i:0..k of its neighbors belong to that class
        self.cond_nprobabilities  = [ [0.] * (k + 1) ] * num_labels
        
        #Computing the prior probabilities P(H_b^l)
        self.compute_prior(instances)
        
        #Computing the posterior probabilities P(E_j^l|H_b^l)
        self.compute_cond(instances)
        
        return MLkNNClassifier(instances = instances, label_indices = self.label_indices, 
                               prior_probabilities = self.prior_probabilities, 
                               prior_nprobabilities = self.prior_nprobabilities,
                               cond_probabilities = self.cond_probabilities,
                               cond_nprobabilities = self.cond_nprobabilities,
                               knn = self.knn,
                               k = self.k,
                               weight_id = weight_id)

    def compute_prior(self,instances):
        """ Computing Prior and PriorN Probabilities for each class of the training set """
        num_instances = len(instances)
        for i in range(self.num_labels):
            temp_ci = 0
            for j in range(num_instances):
                value = instances[j][self.label_indices[i]].value
                if value == '1':
                    temp_ci = temp_ci+1
            self.prior_probabilities[i] = (self.smooth + temp_ci) / (self.smooth * 2 + num_instances)
            self.prior_nprobabilities[i] = 1 - self.prior_probabilities[i]
            
    def compute_cond(self,instances):
        """ Computing Cond and CondN Probabilities for each class of the training set """
        num_labels = self.num_labels
        label_indices = self.label_indices
        k = self.k
        num_instances = len(instances)
        
        temp_ci  = [ [0] * (k + 1) ] * num_labels
        temp_nci = [ [0] * (k + 1) ] * num_labels

        for i  in range(num_instances):
            neighbors = self.knn(instances[i], k)
                 
            # now compute values of temp_ci and temp_nci for every class label
            for j in range(num_labels):
                aces = 0 # num of aces in Knn for j
                for m in range(k):
                    value = neighbors[m].get_class().value[j]
                    if value == '1':
                        aces = aces + 1
                     
                #raise the counter of temp_ci[j][aces] and temp_nci[j][aces] by 1
                if instances[i][label_indices[j]].value == '1':
                    temp_ci[j][aces] = temp_ci[j][aces] + 1
                else:
                    temp_nci[j][aces] = temp_nci[j][aces] + 1
                
        # compute cond_probabilities[i][..] for labels based on temp_ci[]
        for i in range(num_labels):
            temp1 = 0
            temp2 = 0
            for j in range(k + 1):
                temp1 += temp_ci[i][j]
                temp2 += temp_nci[i][j]
            for j in range(k + 1):
                self.cond_probabilities[i][j] = (self.smooth + temp_ci[i][j]) / (self.smooth * (k + 1) + temp1)
                self.cond_nprobabilities[i][j] = (self.smooth + temp_nci[i][j]) / (self.smooth * (k + 1) + temp2)
 
class MLkNNClassifier(_multiknn.MultikNNClassifier):      
    def __call__(self, example, result_type=Orange.classification.Classifier.GetValue):
        num_labels = len(self.label_indices)
        domain = self.instances.domain
        labels = []
        prob = []
        if num_labels == 0:
            raise ValueError, "has no label attribute: 'the multilabel data should have at last one label attribute' "
        
        #Computing y_t and r_t
        neighbors = self.knn(example, self.k)
        for i in range(num_labels):
            # compute sum of aces in KNN
            aces = 0  #num of aces in Knn for i
            for m in range(self.k):
                value = neighbors[m].get_class().value[i]
                if value == '1':
                    aces = aces + 1
    
            prob_in = self.prior_probabilities[i] * self.cond_probabilities[i][aces]
            prob_out = self.prior_nprobabilities[i] * self.cond_nprobabilities[i][aces]
                
            if prob_in > prob_out:
                labels.append(Orange.data.Value(domain[self.label_indices[i]],'1'))
            elif prob_in < prob_out:
                labels.append(Orange.data.Value(domain[self.label_indices[i]],'0'))
            else:
                rnd = random.randint(0,1)
                if rnd == 0:
                    labels.append(Orange.data.Value(domain[self.label_indices[i]],'0'))
                else:
                    labels.append(Orange.data.Value(domain[self.label_indices[i]],'1'))
            
            #ranking function
            prob.append( prob_in / (prob_in + prob_out) )
       
        disc = Orange.statistics.distribution.Discrete(prob)
        disc.variable = Orange.core.EnumVariable(
            values = [domain[val].name for index,val in enumerate(self.label_indices)])
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return disc
        return labels,disc
        
#########################################################################################
# Test the code, run from DOS prompt
# assume the data file is in proper directory

if __name__ == "__main__":
    data = Orange.data.Table("emotions.tab")

    classifier = Orange.multilabel.MLkNNLearner(data,5,1.0)
    for i in range(10):
        c,p = classifier(data[i],Orange.classification.Classifier.GetBoth)
        print c,p