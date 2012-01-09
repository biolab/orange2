"""\
==========================================
Partial least sqaures regression (``PLS``)
==========================================

.. index:: regression

.. _`Parital Least Squares Regression`: http://en.wikipedia.org/wiki/Partial_least_squares_regression

Implementation is based on `Scikit learn python implementation`_

Example ::

    >>> import Orange
    >>>     from Orange.regression import pls
    >>> table = Orange.data.Table("test-pls.tab")
    >>> # set independent and response variables
    >>> x = [var for var in table.domain if var.name[0]=="X"]
    >>> y = [var for var in table.domain if var.name[0]=="Y"]
    >>> print x
        [FloatVariable 'X1', FloatVariable 'X2', FloatVariable 'X3']
    >>> print y
        [FloatVariable 'Y1', FloatVariable 'Y2', FloatVariable 'Y3', FloatVariable 'Y4']
    >>> # The information which variables are independent and which are responses
    >>> # can be provided in the data definition.
    >>> # If a variable var has key "label" in dictionary Orange.data.Domain[var].attributes
    >>> # it is considered as a response variable.
    >>> # In such situation x and y do not need to be specified.
    >>> l = pls.PLSRegressionLearner()
    >>> c = l(table, x_vars=x, y_vars=y)
    >>> c.print_pls_regression_coefficients()
           Y1     Y2     Y3     Y4     
    X1     0.513  0.915  0.341  -0.069  
    X2     0.641  -1.044  0.249  -0.015  
    X3     1.393  0.050  0.729  -0.108  
    >>> 

.. autoclass:: PLSRegressionLearner
    :members:

.. autoclass:: PLSRegression
    :members:

Utility functions
-----------------

.. autofunction:: normalize_matrix

.. autofunction:: nipals_xy

.. autofunction:: svd_xy

.. _`Scikit learn python implementation`: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pls.py

"""

import Orange
import numpy

from Orange.regression import base
from Orange.regression.earth import data_label_mask
from numpy import dot, zeros
from numpy import linalg
from numpy.linalg import svd, pinv

from Orange.misc import deprecated_members, deprecated_keywords


def normalize_matrix(X):
    """ Normalizes matrix, i.e. subtracts column means
    and divides them by column standard deviations.
    Returns the standardized matrix, sample mean and
    standard deviation

    :param X: data matrix
    :type X: :class:`numpy.array`
   
    """
    mu_x, sigma_x = numpy.mean(X, axis=0), numpy.std(X, axis=0)
    sigma_x[sigma_x == 0] = 1.
    return (X - mu_x)/sigma_x, mu_x, sigma_x

@deprecated_keywords({"maxIter": "max_iter"})
def nipals_xy(X, Y, mode="PLS", max_iter=500, tol=1e-06):
    """
    NIPALS algorithm. Returns the first left and rigth singular
    vectors of X'Y.

    :param X, Y: data matrix
    :type X, Y: :class:`numpy.array`

    :param mode: possible values "PLS" (default) or "CCA" 
    :type mode: string

    :param max_iter: maximal number of iterations (default 500)
    :type max_iter: int

    :param tol: tolerance parameter, if norm of difference
        between two successive left singular vectors is less than tol,
        iteration is stopped
    :type tol: a not negative float
            
    """
    yScore, uOld, ite = Y[:, [0]], 0, 1
    Xpinv = Ypinv = None
    # Inner loop of the Wold algo.
    while True and ite < max_iter:
        # Update u: the X weights
        if mode == "CCA":
            if Xpinv is None:
                Xpinv = linalg.pinv(X) # compute once pinv(X)
            u = dot(Xpinv, yScore)
        else: # mode PLS
        # Mode PLS regress each X column on yScore
            u = dot(X.T, yScore) / dot(yScore.T, yScore)
        # Normalize u
        u /= numpy.sqrt(dot(u.T, u))
        # Update xScore: the X latent scores
        xScore = dot(X, u)

        # Update v: the Y weights
        if mode == "CCA":
            if Ypinv is None:
                Ypinv = linalg.pinv(Y) # compute once pinv(Y)
            v = dot(Ypinv, xScore)
        else:
            # Mode PLS regress each X column on yScore
            v = dot(Y.T, xScore) / dot(xScore.T, xScore)
        # Normalize v
        v /= numpy.sqrt(dot(v.T, v))
        # Update yScore: the Y latent scores
        yScore = dot(Y, v)

        uDiff = u - uOld
        if dot(uDiff.T, uDiff) < tol or Y.shape[1] == 1:
            break
        uOld = u
        ite += 1
    return u, v

def svd_xy(X, Y):
    """ Returns the first left and right singular
    vectors of X'Y.

    :param X, Y: data matrix
    :type X, Y: :class:`numpy.array`    
    
    """
    U, s, V = svd(dot(X.T, Y), full_matrices=False)
    u = U[:, [0]]
    v = V.T[:, [0]]
    return u, v


def select_attrs(table, attributes, class_var=None, metas=None):
    """ Select only ``attributes`` from the ``table``.
    """
    domain = Orange.data.Domain(attributes, class_var)
    if metas:
        domain.add_metas(metas)
    return Orange.data.Table(domain, table)


class PLSRegressionLearner(base.BaseRegressionLearner):
    """ Fits the partial least squares regression model,
    i.e. learns the regression parameters. The implementation is based on
    `Scikit learn python implementation`_
    
    The class is derived from
    :class:`Orange.regression.base.BaseRegressionLearner`
    which is used for preprocessing the data (continuization and imputation)
    before fitting the regression parameters
    
    """

    def __init__(self, n_comp=2, deflation_mode="regression", mode="PLS",
                 algorithm="nipals", max_iter=500, 
                 imputer=None, continuizer=None,
                 **kwds):
        """
        .. attribute:: n_comp
    
            number of components to keep. Default: 2

        .. attribute:: deflation_mode
    
            "canonical" or "regression" (default)

        .. attribute:: mode
    
            "CCA" or "PLS" (default)


        .. attribute:: algorithm
    
            The algorithm used to estimate the weights:
            "nipals" or "svd" (default)


        """        
        self.n_comp = n_comp
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.set_imputer(imputer=imputer)
        self.set_continuizer(continuizer=continuizer)
        self.__dict__.update(kwds)

    @deprecated_keywords({"xVars": "x_vars", "yVars": "y_vars"})
    def __call__(self, table, weight_id=None, x_vars=None, y_vars=None):
        """
        :param table: data instances.
        :type table: :class:`Orange.data.Table`

        :param x_vars, y_vars: List of input and response variables
            (`Orange.data.variable.Continuous` or `Orange.data.variable.Discrete`).
            If None (default) it is assumed that data definition provides information
            which variables are reponses and which not. If a variable var
            has key "label" in dictionary Orange.data.Domain[var].attributes
            it is treated as a response variable
        :type x_vars, y_vars: list            

        """     
        domain = table.domain
        if x_vars is None and y_vars is None:
            # Response variables are defined in the table.
            x_vars = domain.features
            if domain.class_var:
                y_vars = [domain.class_var]
                multitarget = False
            elif domain.class_vars:
                y_vars = domain.class_vars
                multitarget = True
            else:
                raise TypeError('Class-less domain (x-vars and y-vars needed).')
            x_table = select_attrs(table, x_vars)
            y_table = select_attrs(table, y_vars)
        elif x_vars and y_vars:
            # independent and response variables are passed by the caller
            multitarget = True
        else:
            raise ValueError("Both x_vars and y_vars must be defined.")

        x_table = select_attrs(table, x_vars)
        y_table = select_attrs(table, y_vars)

        # dicrete values are continuized        
        x_table = self.continuize_table(x_table)
        y_table = self.continuize_table(y_table)
        # missing values are imputed
        x_table = self.impute_table(x_table)
        y_table = self.impute_table(y_table)
        
        # Collect the new transformed x_vars/y_vars 
        x_vars = list(x_table.domain.variables)
        y_vars = list(y_table.domain.variables)
        
        domain = Orange.data.Domain(x_vars + y_vars, False)
        
        x = x_table.to_numpy()[0]
        y = y_table.to_numpy()[0]
        
        kwargs = self.fit(x, y)
        return PLSRegression(domain=domain, x_vars=x_vars, y_vars=y_vars,
                             **kwargs)

    def fit(self, X, Y):
        """ Fits all unknown parameters, i.e.
        weights, scores, loadings (for x and y) and regression coefficients.
        Returns a dict with all of the parameters.
        
        """
        # copy since this will contain the residuals (deflated) matrices

        X, Y = X.copy(), Y.copy()
        if Y.ndim == 1:
            Y = Y.reshape((Y.size, 1))
        n, p = X.shape
        q = Y.shape[1]

        # normalization of data matrices
        X, muX, sigmaX = normalize_matrix(X)
        Y, muY, sigmaY = normalize_matrix(Y)
        # Residuals (deflated) matrices
        Xk, Yk = X, Y
        # Results matrices
        T, U = zeros((n, self.n_comp)), zeros((n, self.n_comp))
        W, C = zeros((p, self.n_comp)), zeros((q, self.n_comp))
        P, Q = zeros((p, self.n_comp)), zeros((q, self.n_comp))      

        # NIPALS over components
        for k in xrange(self.n_comp):
            # Weights estimation (inner loop)
            if self.algorithm == "nipals":
                u, v = nipals_xy(X=Xk, Y=Yk, mode=self.mode, 
                                 max_iter=self.max_iter)
            elif self.algorithm == "svd":
                u, v = svd_xy(X=Xk, Y=Yk)
            # compute scores
            xScore, yScore = dot(Xk, u), dot(Yk, v)
            # Deflation (in place)
            # - regress Xk's on xScore
            xLoadings = dot(Xk.T, xScore) / dot(xScore.T, xScore)
            # - substract rank-one approximations to obtain remainder matrix
            Xk -= dot(xScore, xLoadings.T)
            if self.deflation_mode == "canonical":
                # - regress Yk's on yScore, then substract rank-one approx.
                yLoadings = dot(Yk.T, yScore) / dot(yScore.T, yScore)
                Yk -= dot(yScore, yLoadings.T)
            if self.deflation_mode == "regression":
                # - regress Yk's on xScore, then substract rank-one approx.
                yLoadings = dot(Yk.T, xScore) / dot(xScore.T, xScore)
                Yk -= dot(xScore, yLoadings.T)
            # Store weights, scores and loadings 
            T[:, k] = xScore.ravel() # x-scores
            U[:, k] = yScore.ravel() # y-scores
            W[:, k] = u.ravel() # x-weights
            C[:, k] = v.ravel() # y-weights
            P[:, k] = xLoadings.ravel() # x-loadings
            Q[:, k] = yLoadings.ravel() # y-loadings
        # X = TP' + E and Y = UQ' + E

        # Rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (W* : q x k matrix)
        xRotations = dot(W, pinv(dot(P.T, W)))
        if Y.shape[1] > 1:
            yRotations = dot(C, pinv(dot(Q.T, C)))
        else:
            yRotations = numpy.ones(1)

        if True or self.deflation_mode == "regression":
            # Estimate regression coefficient
            # Y = TQ' + E = X W(P'W)^-1Q' + E = XB + E
            # => B = W*Q' (p x q)
            coefs = dot(xRotations, Q.T)
            coefs = 1. / sigmaX.reshape((p, 1)) * \
                    coefs * sigmaY
        
        return {"mu_x": muX, "mu_y": muY, "sigma_x": sigmaX,
                "sigma_y": sigmaY, "T": T, "U":U, "W":U, 
                "C": C, "P":P, "Q":Q, "x_rotations": xRotations,
                "y_rotations": yRotations, "coefs": coefs}

deprecated_members({"nComp": "n_comp",
                    "deflationMode": "deflation_mode",
                    "maxIter": "max_iter"},
                   wrap_methods=["__init__"],
                   in_place=True)(PLSRegressionLearner)

class PLSRegression(Orange.classification.Classifier):
    """ PLSRegression predicts value of the response variables
    based on the values of independent variables.
    
    Basic notations:
    n - number of data instances
    p - number of independent variables
    q - number of reponse variables

    .. attribute:: T
    
        A n x n_comp numpy array of x-scores

    .. attribute:: U
    
        A n x n_comp numpy array of y-scores

    .. attribute:: W
    
        A p x n_comp numpy array of x-weights

    .. attribute:: C
    
        A q x n_comp numpy array of y-weights

    .. attribute:: P
    
        A p x n_comp numpy array of x-loadings

    .. attribute:: Q
    
        A q x n_comp numpy array of y-loading

    .. attribute:: coefs
    
        A p x q numpy array coefficients
        of the linear model: Y = X coefs + E

    .. attribute:: x_vars
    
        list of independent variables

    .. attribute:: y_vars
    
        list of response variables 
        
    """
    def __init__(self, domain=None, multitarget=False, coefs=None, sigma_x=None, sigma_y=None,
                 mu_x=None, mu_y=None, x_vars=None, y_vars=None, **kwargs):
        self.domain = domain
        self.multitarget = multitarget
        self.coefs = coefs
        self.mu_x, self.mu_y = mu_x, mu_y
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.x_vars, self.y_vars = x_vars, y_vars
            
        for name, val in kwargs.items():
            setattr(self, name, val)

    def __call__(self, instance, result_type=Orange.core.GetValue):
        """
        :param instance: data instance for which the value of the response
            variable will be predicted
        :type instance: :class:`Orange.data.Instance` 
        
        """ 
        instance = Orange.data.Instance(self.domain, instance)
        ins = [instance[v].native() for v in self.x_vars]
        
        if "?" in ins: # missing value -> corresponding coefficient omitted
            def miss_2_0(x): return x if x != "?" else 0
            ins = map(miss_2_0, ins)
        ins = numpy.array(ins)
        xc = (ins - self.mu_x) / self.sigma_x
        predicted = dot(xc, self.coefs) * self.sigma_y + self.mu_y
        y_hat = [var(val) for var, val in zip(self.y_vars, predicted)]
        if result_type == Orange.core.GetValue:
            return y_hat if self.multitarget else y_hat[0]
        else:
            from Orange.statistics.distribution import Distribution
            probs = []
            for var, val in zip(self.y_vars, y_hat):
                dist = Distribution(var)
                dist[val] = 1.0
                probs.append(dist)
            if result_type == Orange.core.GetBoth:
                return (y_hat, probs) if self.multitarget else (y_hat[0], probs[0])
            else:
                return probs if self.multitarget else probs[0]
            
    def to_string(self):
        """ Pretty-prints the coefficient of the PLS regression model.
        """       
        x_vars, y_vars = [x.name for x in self.x_vars], [y.name for y in self.y_vars]
        fmt = "%-6s " + "%-5.3f  " * len(y_vars)
        first = [" " * 7 + "%-6s " * len(y_vars) % tuple(y_vars)]
        lines = [fmt % tuple([x_vars[i]] + list(coef))
                 for i, coef in enumerate(self.coefs)]
        return '\n'.join(first + lines)
            
    def __str__(self):
        return self.to_string()

    """
    def transform(self, X, Y=None):

        # Normalize
        Xc = (X - self.muX) / self.sigmaX
        if Y is not None:
            Yc = (Y - self.muY) / self.sigmaY
        # Apply rotation
        xScores = dot(Xc, self.xRotations)
        if Y is not None:
            yScores = dot(Yc, self.yRotations)
            return xScores, yScores

        return xScores
    """
              
deprecated_members({"xVars": "x_vars", 
                    "yVars": "y_vars",
                    "muX": "mu_x",
                    "muY": "mu_y",
                    "sigmaX": "sigma_x",
                    "sigmaY": "sigma_y"},
                   wrap_methods=["__init__"],
                   in_place=True)(PLSRegression)
                   
if __name__ == "__main__":

    import Orange
    from Orange.regression import pls

    data = Orange.data.Table("test-pls.tab")
    l = pls.PLSRegressionLearner()

    x = [var for var in data.domain.features if var.name[0]=="X"]
    y = [var for var in data.domain.class_vars if var.name[0]=="Y"]
    print x, y
#    c = l(data, x_vars=x, y_vars=y)
    c = l(data)

    print c
