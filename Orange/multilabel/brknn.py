""" 
.. index:: BR-kNN Learner

***************************************
BR-kNN Learner
***************************************

BR-kNN Classification is an adaptation of the kNN algorithm for multi-label classification that 
is conceptually equivalent to using the popular Binary Relevance problem transformation method 
in conjunction with the kNN algorithm. It also implements two extensions of BR-kNN. 
For more information, see E. Spyromitros, G. Tsoumakas, I. Vlahavas, 
`An Empirical Study of Lazy Multilabel Classification Algorithms <http://mlkd.csd.auth.gr/multilabel.html>`_, 
Proc. 5th Hellenic Conference on Artificial Intelligence (SETN 2008), Springer, Syros, Greece, 2008.  

.. index:: BR-kNN Learner
.. autoclass:: Orange.multilabel.BRkNNLearner
   :members:
   :show-inheritance:
 
   :param instances: a table of instances.
   :type instances: :class:`Orange.data.Table`

.. index:: BRkNN Classifier
.. autoclass:: Orange.multilabel.BRkNNClassifier
   :members:
   :show-inheritance:
   
Examples
========

The following example demonstrates a straightforward invocation of
this algorithm (:download:`mlc-classify.py <code/mlc-classify.py>`):

.. literalinclude:: code/mlc-classify.py
   :lines: 6-9

"""
import random
import math

import Orange
import multiknn as _multiknn

class BRkNNLearner(_multiknn.MultikNNLearner):
    """
    Class implementing the BR-kNN learner. 
    
    .. attribute:: k
    
        Number of neighbours. If set to 0 (which is also the default value), 
        the square root of the number of instances is used.
    
    .. attribute:: ext
    
        Extension type. The default is None, means 'Standard BR'; 'a' means
        predicting top ranked label in case of empty prediction set; 'b' means
        predicting top n ranked labels based on size of labelset in neighbours. 
    
    .. attribute:: knn
        
        :class:`Orange.classification.knn.FindNearest` for nearest neighbor search
    
    """
    def __new__(cls, instances = None, k=1, ext = None, weight_id = 0, **argkw):
        """
        Constructor of BRkNNLearner
        
        :param instances: a table of instances.
        :type instances: :class:`Orange.data.Table`
        
        :param k: number of nearest neighbours used in classification
        :type k: int
        
        :param ext: extension type (default value is None which yields
            the Standard BR), values 'a' and 'b' are also possible.
        :type ext: string
        
        :rtype: :class:`BRkNNLearner`
        """
        
        self = _multiknn.MultikNNLearner.__new__(cls, k, **argkw)
        
        if ext not in [None, 'a', 'b']:
            raise ValueError, "Invalid ext value: should be None, 'a' or 'b'."
        self.ext = ext
        
        if instances:
            self.instances = instances
            self.__init__(**argkw)
            return self.__call__(instances,weight_id)
        else:
            return self

    def __call__(self, instances, weight_id = 0, **kwds):
        if not Orange.multilabel.is_multilabel(instances):
            raise TypeError("The given data set is not a multi-label data set.")

        for k in kwds.keys():
            self.__dict__[k] = kwds[k]
        self._build_knn(instances)

        labeling_f = [BRkNNClassifier.get_labels, BRkNNClassifier.get_labels_a,
                      BRkNNClassifier.get_labels_b][ [None, 'a', 'b'].index(self.ext) ]
        
        return BRkNNClassifier(instances = instances,
                               ext = self.ext,
                               knn = self.knn,
                               k = self.k,
                               labeling_f = labeling_f)

class BRkNNClassifier(_multiknn.MultikNNClassifier):
    def __call__(self, instance, result_type=Orange.classification.Classifier.GetValue):
        """
        :rtype: a list of :class:`Orange.data.Value`, a list of :class:`Orange.statistics.distribution.Distribution`, or a tuple with both
        """
        domain = self.instances.domain

        neighbours = self.knn(instance, self.k)
        
        prob = self.get_prob(neighbours)
        
        labels = self.labeling_f(self, prob, neighbours)
        
        if result_type == Orange.classification.Classifier.GetValue:
            return labels

        dists = [Orange.statistics.distribution.Discrete([1-p, p]) for p in prob]
        for v, d in zip(self.instances.domain.class_vars, dists):
            d.variable = v

        if result_type == Orange.classification.Classifier.GetProbabilities:
            return dists
        return labels, dists
    
    def get_prob(self, neighbours):
        """
        Calculates the probabilities of the labels, based on the neighboring
        instances.
     
        :param neighbours: a list of nearest neighboring instances.
        :type neighbours: list of :class:`Orange.data.Instance`
        
        :rtype: the prob of the labels
        
        """
        total = 0
        label_count = len(self.instances.domain.class_vars)
        confidences = [1.0 / max(1, len(self.instances))] * label_count

        total = float(label_count) / max(1, len(self.instances))
        
        for neigh in neighbours:
            vals = neigh.get_classes()
            for j, value in enumerate(vals):
                if value == '1':
                    confidences[j] += 1
            total += 1

        #Normalize distribution
        if total > 0:
            confidences = [con/total for con in confidences]
        
        return confidences
    
    def get_labels(self, prob, _neighs=None, thresh=0.5):
        return [Orange.data.Value(lvar, str(int(p>thresh)))
                for p, lvar in zip(prob, self.instances.domain.class_vars)]
    
    def get_labels_a(self, prob, _neighs=None):
        """
        used for BRknn-a
        
        :param prob: the probabilities of the labels
        :type prob: list of double
        
        :rtype: the list label value
        """
        labels = self.get_labels(prob)
            
        #assign the class with the greatest confidence
        if all(l.value=='0' for l in labels):
            index = max((v,i) for i,v in enumerate(prob))[1]
            labels[index].value = '1'
        
        return labels
    
    def get_labels_b(self, prob, neighs):
        """
        used for BRknn-b
        
        :param prob: the probabilities of the labels
        :type prob: list of double
        
        :rtype: the list label value
        """
        
        labels = [Orange.data.Value(lvar, '0')
                  for p, lvar in zip(prob, self.instances.domain.class_vars)]
        
        avg_label_cnt = sum(sum(l.value=='1' for l in n.get_classes())
                            for n in neighs) / float(len(neighs))
        avg_label_cnt = int(round(avg_label_cnt))
        
        for p, lval in sorted(zip(prob, labels), reverse=True)[:avg_label_cnt]:
            lval.value = '1'

        return labels
    
#########################################################################################
# Test the code, run from DOS prompt
# assume the data file is in proper directory

if __name__ == "__main__":
    data = Orange.data.Table("emotions.tab")

    classifier = Orange.multilabel.BRkNNLearner(data,5)
    for i in range(10):
        c,p = classifier(data[i],Orange.classification.Classifier.GetBoth)
        print c,p