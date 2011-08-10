""" 
.. index:: BR-kNN Learner

***************************************
BR-kNN Learner
***************************************

BR-kNN Classification is an adaptation of the kNN algorithm for multi-label classification that 
is conceptually equivalent to using the popular Binary Relevance problem transformation method 
in conjunction with the kNN algorithm. It also implements two extensions of BR-kNN. 
For more information, see E. Spyromitros, G. Tsoumakas, I. Vlahavas, 
'An Empirical Study of Lazy Multilabel Classification Algorithms <http://mlkd.csd.auth.gr/multilabel.html>', 
Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008), Springer, Syros, Greece, 2008.  

.. index:: BR-kNN Learner
.. autoclass:: Orange.multilabel.BRkNNLearner
   :members:
   :show-inheritance:
 
   .. method:: __new__(instances, **argkw) 
   BRkNNLearner Constructor
   
   :param instances: a table of instances, covered by the rule.
   :type instances: :class:`Orange.data.Table`

.. index:: BRkNN Classifier
.. autoclass:: Orange.multilabel.BRkNNClassifier
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
import math

class BRkNNLearner(_multiknn.MultikNNLearner):
    """
    Class implementing the BR-kNN algorithm. 
    
    .. attribute:: k
    
        Number of neighbours. If set to 0 (which is also the default value), 
        the square root of the number of instances is used.
    
    .. attribute:: ext
    
        Extension type. The default is None, means 'Standard BR'; 'a' means Predict top ranked label in case of empty prediction set;
        'b' means Predict top n ranked labels based on size of labelset in neighbours 
    
    .. attribute:: knn
        
        :class:`Orange.classification.knn.FindNearest` for nearest neighbor search
    
    """
    def __new__(cls, instances = None, k=1, ext = None, **argkw):
        """
        Constructor of BRkNNLearner
        
        :param instances: a table of instances, covered by the rule.
        :type instances: :class:`Orange.data.Table`
        
        :param k: number of nearest neighbours used in classification
        :type k: int
        
        :param ext:  Extension type (Default value is set to '0' which yields the Standard BR).
        :type smooth: string
        
        :rtype: :class:`BRkNNLearner`
        """
        
        self = _multiknn.MultikNNLearner.__new__(cls, k, **argkw)
        
        if ext and ext <>'a' and ext <> 'b':
            raise ValueError, "invalid ext value: 'the extension value should be only None, 'a' or 'b' "
        self.ext = ext
        
        if instances:
            self.instances = instances
            self.__init__(**argkw)
            return self.__call__(instances)
        else:
            return self

    def __call__(self, instances, **kwds):
        for k in kwds.keys():
            self.__dict__[k] = kwds[k]

        _multiknn.MultikNNLearner.transfor_table(self,instances)
        
        return BRkNNClassifier(instances = instances, label_indices = self.label_indices,
                               ext = self.ext,
                               knn = self.knn,
                               weight_id = self.weight_id,
                               k = self.k)

def max(x,y):
    if x > y:
        return x
    else:
        return y

class BRkNNClassifier(_multiknn.MultikNNClassifier):    
    def __call__(self, example, result_type=Orange.classification.Classifier.GetValue):
        self.num_labels = len(self.label_indices)
        domain = self.instances.domain
        labels = []
        if self.num_labels == 0:
            raise ValueError, "has no label attribute: 'the multilabel data should have at last one label attribute' "

        neighbours = self.knn(example, self.k)
        distances = [inst.get_weight(self.weight_id) for i,inst in enumerate(neighbours)]
        
        prob = self.get_prob(neighbours, distances)
        
        if self.ext == None:
            labels = self.get_label(prob, 0.5)
        elif self.ext == 'a':
            labels = self.get_label_a(prob)
        elif self.ext == 'b':
            labels = self.get_label_b(prob)
        
        disc = Orange.statistics.distribution.Discrete(prob)
        disc.variable = Orange.core.EnumVariable(
            values = [domain[val].name for index,val in enumerate(self.label_indices)])
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels
        if result_type == Orange.classification.Classifier.GetProbabilities:
            return disc
        return labels,disc
    
    def get_prob(self, neighbours, distances):
        """
        Calculates the probabilities of the labels, based on the neighboring instances
     
        :param neighbours: a list of nearest neighboring instances
        :type neighbours: list of :class:`Orange.data.Instance`
        
        :param distances: distance of the neighbours
        :type distances: list of double
        
        :rtype: the prob of the labels
        
        """
        total = 0
        weight = 0
        neighborLabels = 0
        confidences = [0.0]* self.num_labels

        #Set up a correction to the estimator
        for i  in range(self.num_labels):
            confidences[i] = 1.0 / max(1, len(self.instances))
        
        total = self.num_labels / max(1, len(self.instances))
        
        for i in range(self.k):
            #Collect class counts
            current = neighbours[i]
            distances[i] = distances[i] * distances[i]
            distances[i] = math.sqrt(distances[i] / (len(self.instances.domain.variables) - self.num_labels))
           
            weight = 1.0
            #weight *= current.weight();

            for j in range(self.num_labels):
                value = current.get_class().value[j]
                if value == '1':
                    confidences[j] += weight
                    neighborLabels += weight
            total += weight

        self.avgPredictedLabels = math.ceil(neighborLabels / total)
        
        #Normalise distribution
        if total > 0:
            confidences = [con/total for con in confidences]
        
        return confidences
    
    def get_label(self, prob, thresh):
        labels = []
        for i in range(self.num_labels):
            if prob[i] >= thresh:
                labels.append(Orange.data.Value(self.instances.domain[self.label_indices[i]],'1'))
            else:
                labels.append(Orange.data.Value(self.instances.domain[self.label_indices[i]],'0'))
           
        return labels
    
    def get_label_a(self, prob):
        """
        used for BRknn-a
        
        :param prob: the probabilities of the labels
        :type prob: list of double
        
        :rtype: the list label value
        """
        labels = []
        flag = False; #check the case that no label is true

        for i in range(self.num_labels):
            if prob[i] >= 0.5:
                labels.append(Orange.data.Value(self.instances.domain[self.label_indices[i]],'1'))
            else:
                labels.append(Orange.data.Value(self.instances.domain[self.label_indices[i]],'0'))
            
        #assign the class with the greater confidence
        if flag == False:
            max_p = -1
            index = -1
            for i in range(len(prob)):
                if max_p <= prob[i]:
                    max_p = prob[i]
                    index = i
            if index <> -1:
                labels[index].value = '1'
        
        return labels
    
    def get_label_b(self, prob):
        """
        used for BRknn-b
        
        :param prob: the probabilities of the labels
        :type prob: list of double
        
        :rtype: the list label value
        """
        
        labels = []
        for i in range(self.num_labels):
            labels.append(Orange.data.Value(self.instances.domain[self.label_indices[i]],'0'))
        
        prob_copy = prob
        prob_copy.sort()
        
        indices = []
        counter = 0

        for i in range(self.num_labels):
            if prob[i] > prob[self.num_labels - self.avgPredictedLabels]:
                labels[i].value = '1'
                counter = counter + 1
            elif prob[i] == prob[self.num_labels - self.avgPredictedLabels]:
                indices.append(i)

        size = len(indices)

        j = avgPredictedLabels - counter
        while j > 0:
            next = rrandom.randint(0,size-1)
            if labels[indices[next]] <> '1':
                labels[indices[next]] = '1'
                j = j - 1
        
        return labels       
    