""" 
*******************************
Gaussian Mixtures (``mixture``)
*******************************

This module implements a Gaussian mixture model.

Example ::
    
    >>> mixture = GaussianMixture(data, n=3)
    >>> print mixture.means
    >>> print mixture.weights
    >>> print mixture.covariances
    >>> plot_model(data, mixture, samples=40)
    
"""

import sys, os
import numpy
import random
import Orange

class GMModel(object):
    """ Gaussian mixture model
    """
    def __init__(self, weights, means, covariances, inv_covariances=None,
                 cov_determinants=None):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        if inv_covariances is None:
            self.inv_covariances = [numpy.linalg.pinv(cov) for cov in covariances]
        else:
            self.inv_covariances = inv_covariances
            
        if cov_determinants is None:
            self.cov_determinants = [numpy.linalg.det(cov) for cov in covariances]
        else:
            self.cov_determinants = cov_determinants
        
        
    def __call__(self, instance):
        """ Return the probability of instance.
        """
        return numpy.sum(prob_est([instance], self.weights, self.means,
                                  self.covariances,
                                  self.inv_covariances,
                                  self.cov_determinants))
        
    def __getitem__(self, index):
        """ Return the index-th gaussian.
        """ 
        return GMModel([1.0], self.means[index: index + 1],
                       self.covariances[index: index + 1],
                       self.inv_covariances[index: index + 1],
                       self.cov_determinants[index: index + 1])

    def __len__(self):
        return len(self.weights)
    
    
def init_random(data, n, *args, **kwargs):
    """ Init random means and correlations from a data table.
    
    :param data: data table
    :type data: :class:`Orange.data.Table`
    :param n: Number of centers and correlations to return.
    :type n: int
    
    """
    if isinstance(data, Orange.data.Table):
        array, w, c = data.toNumpyMA()
    else:
        array = numpy.asarray(data)
        
    min, max = array.max(0), array.min(0)
    dim = array.shape[1]
    means = numpy.zeros((n, dim))
    for i in range(n):
        means[i] = [numpy.random.uniform(low, high) for low, high in zip(min, max)]
        
    correlations = [numpy.asmatrix(numpy.eye(dim)) for i in range(n)]
    return means, correlations

def init_kmeans(data, n, *args, **kwargs):
    """ Init with k-means algorithm.
    
    :param data: data table
    :type data: :class:`Orange.data.Table`
    :param n: Number of centers and correlations to return.
    :type n: int
    
    """
    if not isinstance(data, Orange.data.Table):
        raise TypeError("Orange.data.Table instance expected!")
    from Orange.clustering.kmeans import Clustering
    km = Clustering(data, centroids=n, maxiters=20, nstart=3)
    centers = Orange.data.Table(km.centroids)
    centers, w, c = centers.toNumpyMA()
    dim = len(data.domain.attributes)
    correlations = [numpy.asmatrix(numpy.eye(dim)) for i in range(n)]
    return centers, correlations
    
def prob_est1(data, mean, covariance, inv_covariance=None, det=None):
    """ Return the probability of data given mean and covariance matrix
    """
    data = numpy.asmatrix(data)
    mean = numpy.asmatrix(mean) 
    if inv_covariance is None:
        inv_covariance = numpy.linalg.pinv(covariance)
        
    inv_covariance = numpy.asmatrix(inv_covariance)
    
    diff = data - mean
    p = numpy.zeros(data.shape[0])
    for i in range(data.shape[0]):
        d = diff[i]
        p[i] = d * inv_covariance * d.T
        
    p *= -0.5
    p = numpy.exp(p)
    p /= numpy.power(2 * numpy.pi, numpy.rank(covariance) / 2.0)
    if det is None:
        det = numpy.linalg.det(covariance)
    assert(det != 0.0)
    p /= det
    return p


def prob_est(data, weights, means, covariances, inv_covariances=None, cov_determinants=None):
    """ Return the probability estimation of data for each
    gausian given weights, means and covariances.
      
    """
    if inv_covariances is None:
        inv_covariances = [numpy.linalg.pinv(cov) for cov in covariances]
        
    if cov_determinants is None:
        cov_determinants = [numpy.linalg.det(cov) for cov in covariances]
        
    data = numpy.asmatrix(data)
    probs = numpy.zeros((data.shape[0], len(weights)))
    
    for i, (w, mean, cov, inv_cov, det) in enumerate(zip(weights, means,
                                        covariances, inv_covariances,
                                        cov_determinants)):
        probs[:, i] = w * prob_est1(data, mean, cov, inv_cov, det)
        
    return probs

    
class EMSolver(object):
    """ An EM solver for gaussian mixture model
    """
    _TRACE_MEAN = False
    def __init__(self, data, weights, means, covariances):
        self.data = data
        self.weights = weights 
        self.means = means
        self.covariances = covariances
        self.inv_covariances = [numpy.matrix(numpy.linalg.pinv(cov)) for cov in covariances]
        self.cov_determinants = [numpy.linalg.det(cov) for cov in self.covariances]
        self.n_clusters = len(self.weights)
        self.data_dim = self.data.shape[1]
        
        self.probs = prob_est(data, weights, means, covariances,
                              self.inv_covariances, self.cov_determinants)
        
        self.log_likelihood = self._log_likelihood()
#        print "W", self.weights
#        print "P", self.probs
#        print "L", self.log_likelihood 
#        print "C", self.covariances
#        print "Det", self.cov_determinants
        
    def _log_likelihood(self):
        """ Compute the log likelihood of the current solution.
        """
        log_p = numpy.log(numpy.sum(self.weights * self.probs, axis=0))
        return 1.0 / len(self.data) * numpy.sum(log_p)
        
    def E_step(self):
        """ E Step
        """
        self.probs = prob_est(self.data, self.weights, self.means,
                         self.covariances, self.inv_covariances)
        
#        print "PPP", self.probs
#        print "P sum", numpy.sum(self.probs, axis=1).reshape((-1, 1))
        self.probs /= numpy.sum(self.probs, axis=1).reshape((-1, 1))
        
        # Update the Q
#        self.Q = 0.0
#        prob_sum = numpy.sum(self.probs, axis=0)
#        self.Q = sum([p*(numpy.log(w) - 0.5 * numpy.linalg.det(cov)) \
#                      for p, w, cov in zip(prob_sum, self.weights,
#                                           self.covariances)]) * \
#                      len(self.data)
#        
#        for i in range(len(data)):
#            for j in range(self.n_clusters):
#                diff = numpy.asmatrix(self.data[i] - self.means[j])
#                self.Q += - 0.5 * self.probs[i, j] * diff.T * self.inv_covariances[j] * diff
#        print self.Q
                
        self.log_likelihood = self._log_likelihood()
#        print "Prob:", self.probs
#        print "Log like.:", self.log_likelihood
        
    def M_step(self):
        """ M step
        """
        # Compute the new weights
        prob_sum = numpy.sum(self.probs, axis=0)
        weights = prob_sum / numpy.sum(prob_sum)
        
        # Compute the new means
        means = []
        for j in range(self.n_clusters):
            means.append(numpy.sum(self.probs[:, j].reshape((-1, 1)) * self.data, axis=0) /  prob_sum[j]) 
        
        # Compute the new covariances
        covariances = []
        cov_determinants = []
        for j in range(self.n_clusters):
            cov = numpy.zeros(self.covariances[j].shape)
            diff = self.data - means[j]
            diff = numpy.asmatrix(diff)
            for i in range(len(self.data)): # TODO: speed up
                cov += self.probs[i, j] * diff[i].T * diff[i]
                
            cov *= 1.0 / prob_sum[j]
            det = numpy.linalg.det(cov)
            
            covariances.append(numpy.asmatrix(cov))
            cov_determinants.append(det)
#            self.inv_covariances[j] = numpy.linalg.pinv(cov)
#            self.cov_determinants[j] = det
        self.weights = weights
        self.means = numpy.asmatrix(means)
        self.covariances = covariances
        self.cov_determinants = cov_determinants
        
    def one_step(self):
        """ One iteration of both M and E step.
        """
        self.E_step()
        self.M_step()
        
    def run(self, max_iter=sys.maxint, eps=1e-5):
        """ Run the EM algorithm.
        """
        if self._TRACE_MEAN:
            from pylab import plot, show, draw, ion
            ion()
            plot(self.data[:, 0], self.data[:, 1], "ro")
            vec_plot = plot(self.means[:, 0], self.means[:, 1], "bo")[0]
        
        curr_iter = 0
        
        while True:
            old_objective = self.log_likelihood
            self.one_step()
            
            if self._TRACE_MEAN:
                vec_plot.set_xdata(self.means[:, 0])
                vec_plot.set_ydata(self.means[:, 1])
                draw()
            
            curr_iter += 1
#            print curr_iter
#            print abs(old_objective - self.log_likelihood)
            if abs(old_objective - self.log_likelihood) < eps or curr_iter > max_iter:
                break    
    
class GaussianMixture(object):
    """ Computes the gaussian mixture model from an Orange data-set.
    """
    def __new__(cls, data=None, weight_id=None, **kwargs):
        self = object.__new__(cls)
        if data is not None:
            self.__init__(**kwargs)
            return self.__call__(data, weight_id)
        else:
            return self
        
    def __init__(self, n=3, init_function=init_kmeans):
        self.n = n
        self.init_function = init_function
        
    def __call__(self, data, weight_id=None):
        from Orange.preprocess import Preprocessor_impute, DomainContinuizer
#        data = Preprocessor_impute(data)
        dc = DomainContinuizer()
        dc.multinomial_treatment = DomainContinuizer.AsOrdinal
        dc.continuous_treatment = DomainContinuizer.NormalizeByVariance
        dc.class_treatment = DomainContinuizer.Ignore
        domain = dc(data)
        data = data.translate(domain)
        
        means, correlations = self.init_function(data, self.n)
        means = numpy.asmatrix(means)
        array, _, _ = data.to_numpy_MA()
#        avg = numpy.mean(array, axis=0)
#        array -= avg.reshape((1, -1))
#        means -= avg.reshape((1, -1))
#        std = numpy.std(array, axis=0)
#        array /= std.reshape((1, -1))
#        means /= std.reshape((1, -1))
        solver = EMSolver(array, numpy.ones((self.n)) / self.n,
                          means, correlations)
        solver.run()
        norm_model = GMModel(solver.weights, solver.means, solver.covariances)
        return GMClusterModel(domain, norm_model)
    
        
class GMClusterModel(object):
    """ 
    """
    def __init__(self, domain, norm_model):
        self.domain = domain
        self.norm_model = norm_model
        self.cluster_vars = [Orange.feature.Continuous("cluster %i" % i)\
                             for i in range(len(norm_model))]
        self.weights = self.norm_model.weights
        self.means = self.norm_model.means
        self.covariances = self.norm_model.covariances
        self.inv_covariances = self.norm_model.inv_covariances
        self.cov_determinants = self.norm_model.cov_determinants
        
    def __call__(self, instance, *args):
        data = Orange.data.Table(self.domain, [instance])
        data,_,_ = data.to_numpy_MA()
#        print data
        
        p = prob_est(data, self.norm_model.weights,
                     self.norm_model.means,
                     self.norm_model.covariances,
                     self.norm_model.inv_covariances,
                     self.norm_model.cov_determinants)
#        print p
        p /= numpy.sum(p)
        vals = []
        for p, var in zip(p[0], self.cluster_vars):
            vals.append(var(p))
        return vals
        
        
def plot_model(data_array, mixture, axis=(0, 1), samples=20, contour_lines=20):
    """ Plot the scaterplot of data_array and the contour lines of the
    probability for the mixture.
     
    """
    import matplotlib
    import matplotlib.pylab as plt
    import matplotlib.cm as cm
    
    axis = list(axis)
    
    if isinstance(mixture, GMClusterModel):
        mixture = mixture.norm_model
    
    if isinstance(data_array, Orange.data.Table):
        data_array, _, _ = data_array.to_numpy_MA()
    array = data_array[:, axis]
    
    weights = mixture.weights
    means = mixture.means[:, axis]
    
    covariances = [cov[axis,:][:, axis] for cov in mixture.covariances] # TODO: Need the whole marginal distribution. 
    
    gmm = GMModel(weights, means, covariances)
    
    min = numpy.min(array, 0)
    max = numpy.max(array, 0)
    extent = (min[0], max[0], min[1], max[1])
    
    X = numpy.linspace(min[0], max[0], samples)
    Y = numpy.linspace(min[1], max[1], samples)
    
    Z = numpy.zeros((X.shape[0], Y.shape[0]))
    
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            Z[i, j] = gmm([x, y])
            
    plt.plot(array[:,0], array[:,1], "ro")
    plt.contour(X, Y, Z.T, contour_lines,
                extent=extent)
    
    im = plt.imshow(Z.T, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=extent)
    
    plt.plot(means[:, 0], means[:, 1], "b+")
    plt.show()
    
def test(seed=0):
#    data = Orange.data.Table(os.path.expanduser("brown-selected.tab"))
#    data = Orange.data.Table(os.path.expanduser("~/Documents/brown-selected-fss.tab"))
#    data = Orange.data.Table(os.path.expanduser("~/Documents/brown-selected-fss-1.tab"))
#    data = Orange.data.Table(os.path.expanduser("~/ozone1"))
    data = Orange.data.Table("iris.tab")
#    data = Orange.data.Table(Orange.data.Domain(data.domain[:2], None), data)
    numpy.random.seed(seed)
    random.seed(seed)
    gmm = GaussianMixture(data, n=3, init_function=init_kmeans)
    data = data.translate(gmm.domain)
    plot_model(data, gmm, axis=(0, 2), samples=40, contour_lines=100)

    
    
if __name__ == "__main__":
    test()
    
    
