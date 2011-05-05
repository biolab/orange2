""" Implements a Gaussian mixture model.

Example ::
    
    >>> mixture = GaussianMixture(data, n_centers=3)
    >>> print mixture.means
    >>> print mixture.weights
    >>> print mixture.covariances
    >>> plot_model(data, mixture, samples=40)
    
"""

import sys, os
import numpy
import Orange.data

class GMModel(object):
    """ Gaussian mixture model
    """
    def __init__(self, weights, means, covariances):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.inverse_covariances = [numpy.linalg.pinv(cov) for cov in covariances]
        
    def __call__(self, instance):
        """ Return the conditional probability of instance.
        """
        return numpy.sum(prob_est([instance], self.weights, self.means, self.covariances))
        
    def __getitem__(self, index):
        """ Return the index-th gaussian
        """ 
        return GMModel([1.0], self.means[index: index + 1], self.covariances[index: index + 1])
    
#    def __getslice__(self, slice):
#        pass

    def __len__(self):
        return len(self.weights)
    
    
def init_random(array, n_centers, *args, **kwargs):
    """ Init random means
    """
    min, max = array.max(0), array.min(0)
    dim = array.shape[1]
    means = numpy.zeros((n_centers, dim))
    for i in range(n_centers):
        means[i] = [numpy.random.uniform(low, high) for low, high in zip(min, max)]
        
    correlations = [numpy.asmatrix(numpy.eye(dim)) for i in range(n_centers)]
    return means, correlations
    
    
def prob_est1(data, mean, covariance, inv_covariance=None):
    """ Return the probability of data given mean and covariance matrix 
    """
    data = numpy.asmatrix(data)
     
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
    det = numpy.linalg.det(covariance)
    assert(det != 0.0)
    p /= det
#    if det != 0.0:
#        p /= det
#    else:
#        p = numpy.ones(p.shape) / p.shape[0]
    return p


def prob_est(data, weights, means, covariances, inv_covariances=None):
    """ Return the probability estimation of data given weighted, means and
    covariances.
      
    """
    if inv_covariances is None:
        inv_covariances = [numpy.linalg.pinv(cov) for cov in covariances]
        
    data = numpy.asmatrix(data)
    probs = numpy.zeros((data.shape[0], len(weights)))
    
    for i, (w, mean, cov, inv_cov) in enumerate(zip(weights, means, covariances, inv_covariances)):
        probs[:, i] = w * prob_est1(data, mean, cov, inv_cov)
        
    return probs

    
class EMSolver(object):
    """ An EM solver for gaussian mixture model
    """
    def __init__(self, data, weights, means, covariances):
        self.data = data
        self.weights = weights 
        self.means = means
        self.covariances = covariances
        self.inv_covariances = [numpy.matrix(numpy.linalg.pinv(cov)) for cov in covariances]
        
        self.n_clusters = len(self.weights)
        self.data_dim = self.data.shape[1]
        
        self.probs = prob_est(data, weights, means, covariances, self.inv_covariances)
        
        self.log_likelihood = self._log_likelihood()
        
        
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
        
        
    def M_step(self):
        """ M step
        """
        # Update the weights
        prob_sum = numpy.sum(self.probs, axis=0)
        
        self.weights = prob_sum / numpy.sum(prob_sum)
        
        # Update the means
        for j in range(self.n_clusters):
            self.means[j] = numpy.sum(self.probs[:, j].reshape((-1, 1)) * self.data, axis=0) /  prob_sum[j] 
        
        # Update the covariances
        for j in range(self.n_clusters):
            cov = numpy.zeros(self.covariances[j].shape)
            diff = self.data - self.means[j]
            diff = numpy.asmatrix(diff)
            for i in range(len(self.data)): # TODO: speed up
                cov += self.probs[i, j] * diff[i].T * diff[i]
                
            cov *= 1.0 / prob_sum[j]
            self.covariances[j] = cov
            self.inv_covariances[j] = numpy.linalg.pinv(cov)
        
    def one_step(self):
        """ One iteration of both M and E step.
        """
        self.E_step()
        self.M_step()
        
    def run(self, max_iter=sys.maxint, eps=1e-5):
        """ Run the EM algorithm.
        """
        
#        from pylab import plot, show, draw, ion
#        ion()
#        plot(self.data[:, 0], self.data[:, 1], "ro")
#        vec_plot = plot(self.means[:, 0], self.means[:, 1], "bo")[0]
        curr_iter = 0
        
        while True:
            old_objective = self.log_likelihood
            self.one_step()
            
#            vec_plot.set_xdata(self.means[:, 0])
#            vec_plot.set_ydata(self.means[:, 1])
#            draw()
            
            curr_iter += 1
            print curr_iter
            print abs(old_objective - self.log_likelihood)
            if abs(old_objective - self.log_likelihood) < eps or curr_iter > max_iter:
                break
        
        
class GASolver(object):
    """ A toy genetic algorithm solver 
    """
    def __init__(self, data, weights, means, covariances):
        raise NotImplementedError


class PSSolver(object):
    """ A toy particle swarm solver
    """
    def __init__(self, data, weights, means, covariances):
        raise NotImplementedError

class HybridSolver(object):
    """ A hybrid solver
    """
    def __init__(self, data, weights, means, covariances):
        raise NotImplementedError
    
    
class GaussianMixture(object):
    def __new__(cls, data=None, weightId=None, **kwargs):
        self = object.__new__(cls)
        if data is not None:
            self.__init__(**kwargs)
            return self.__call__(data, weightId)
        else:
            return self
        
    def __init__(self, n_centers=3, init_function=init_random):
        self.n_centers = n_centers
        self.init_function = init_function
        
    def __call__(self, data, weightId=None):
        array, _, _ = data.to_numpy_MA()
        solver = EMSolver(array, numpy.ones((self.n_centers)) / self.n_centers,
                          *self.init_function(array, self.n_centers))
        solver.run()
        return GMModel(solver.weights, solver.means, solver.covariances)
        
        
def plot_model(data_array, mixture, axis=(0, 1), samples=20, contour_lines=20):
    
    import matplotlib
    import matplotlib.pylab as plt
    import matplotlib.cm as cm
    
    axis = list(axis)
    if isinstance(data_array, Orange.data.Table):
        data_array, _, _ = data_array.to_numpy_MA()
    array = data_array[:, axis]
    
    weights = mixture.weights
    means = [m[axis] for m in mixture.means]
    
    covariances = [cov[axis,:][:, axis] for cov in mixture.covariances] 
    
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
    
    plt.show()
    
def test():
#    data = Orange.data.Table(os.path.expanduser("../../doc/datasets/brown-selected.tab"))
    data = Orange.data.Table(os.path.expanduser("~/Documents/brown-selected-fss.tab"))
#    data = Orange.data.Table("../../doc/datasets/iris.tab")
#    data = Orange.data.Table(Orange.data.Domain(data.domain[:2], None), data)
    numpy.random.seed(0)
    gmm = GaussianMixture(data, n_centers=3)
    plot_model(data, gmm, axis=(0,1), samples=40, contour_lines=20)

    
    
if __name__ == "__main__":
    test()
    
    
