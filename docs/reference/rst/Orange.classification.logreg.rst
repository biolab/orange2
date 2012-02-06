.. automodule:: Orange.classification.logreg

.. index: logistic regression
.. index:
   single: classification; logistic regression

********************************
Logistic regression (``logreg``)
********************************

`Logistic regression <http://en.wikipedia.org/wiki/Logistic_regression>`_
is a statistical classification methods that fits data to a logistic
function. Orange's implementation of algorithm
can handle various anomalies in features, such as constant variables and
singularities, that could make direct fitting of logistic regression almost
impossible. Stepwise logistic regression, which iteratively selects the most
informative features, is also supported.

.. autoclass:: LogRegLearner
   :members:

.. class :: LogRegClassifier

    A logistic regression classification model. Stores estimated values of
    regression coefficients and their significances, and uses them to predict
    classes and class probabilities.

    .. attribute :: beta

        Estimated regression coefficients.

    .. attribute :: beta_se

        Estimated standard errors for regression coefficients.

    .. attribute :: wald_Z

        Wald Z statistics for beta coefficients. Wald Z is computed
        as beta/beta_se.

    .. attribute :: P

        List of P-values for beta coefficients, that is, the probability
        that beta coefficients differ from 0.0. The probability is
        computed from squared Wald Z statistics that is distributed with
        Chi-Square distribution.

    .. attribute :: likelihood

        The probability of the sample (ie. learning examples) observed on
        the basis of the derived model, as a function of the regression
        parameters.

    .. attribute :: fit_status

        Tells how the model fitting ended - either regularly
        (:obj:`LogRegFitter.OK`), or it was interrupted due to one of beta
        coefficients escaping towards infinity (:obj:`LogRegFitter.Infinity`)
        or since the values didn't converge (:obj:`LogRegFitter.Divergence`). The
        value tells about the classifier's "reliability"; the classifier
        itself is useful in either case.

    .. method:: __call__(instance, result_type)

        Classify a new instance.

        :param instance: instance to be classified.
        :type instance: :class:`~Orange.data.Instance`
        :param result_type: :class:`~Orange.classification.Classifier.GetValue` or
              :class:`~Orange.classification.Classifier.GetProbabilities` or
              :class:`~Orange.classification.Classifier.GetBoth`

        :rtype: :class:`~Orange.data.Value`,
              :class:`~Orange.statistics.distribution.Distribution` or a
              tuple with both


.. class:: LogRegFitter

    :obj:`LogRegFitter` is the abstract base class for logistic fitters. It
    defines the form of call operator and the constants denoting its
    (un)success:

    .. attribute:: OK

        Fitter succeeded to converge to the optimal fit.

    .. attribute:: Infinity

        Fitter failed due to one or more beta coefficients escaping towards infinity.

    .. attribute:: Divergence

        Beta coefficients failed to converge, but none of beta coefficients escaped.

    .. attribute:: Constant

        There is a constant attribute that causes the matrix to be singular.

    .. attribute:: Singularity

        The matrix is singular.


    .. method:: __call__(examples, weight_id)

        Performs the fitting. There can be two different cases: either
        the fitting succeeded to find a set of beta coefficients (although
        possibly with difficulties) or the fitting failed altogether. The
        two cases return different results.

        `(status, beta, beta_se, likelihood)`
            The fitter managed to fit the model. The first element of
            the tuple, result, tells about the problems occurred; it can
            be either :obj:`OK`, :obj:`Infinity` or :obj:`Divergence`. In
            the latter cases, returned values may still be useful for
            making predictions, but it's recommended that you inspect
            the coefficients and their errors and make your decision
            whether to use the model or not.

        `(status, attribute)`
            The fitter failed and the returned attribute is responsible
            for it. The type of failure is reported in status, which
            can be either :obj:`Constant` or :obj:`Singularity`.

        The proper way of calling the fitter is to expect and handle all
        the situations described. For instance, if fitter is an instance
        of some fitter and examples contain a set of suitable examples,
        a script should look like this::

            res = fitter(examples)
            if res[0] in [fitter.OK, fitter.Infinity, fitter.Divergence]:
               status, beta, beta_se, likelihood = res
               < proceed by doing something with what you got >
            else:
               status, attr = res
               < remove the attribute or complain to the user or ... >


.. class :: LogRegFitter_Cholesky

    The sole fitter available at the
    moment. It is a C++ translation of `Alan Miller's logistic regression
    code <http://users.bigpond.net.au/amiller/>`_. It uses Newton-Raphson
    algorithm to iteratively minimize least squares error computed from
    learning examples.


.. autoclass:: StepWiseFSS
   :members:
   :show-inheritance:

.. autofunction:: dump



Examples
--------

The first example shows a very simple induction of a logistic regression
classifier (:download:`logreg-run.py <code/logreg-run.py>`).

.. literalinclude:: code/logreg-run.py

Result::

    Classification accuracy: 0.778282598819

    class attribute = survived
    class values = <no, yes>

        Attribute       beta  st. error     wald Z          P OR=exp(beta)

        Intercept      -1.23       0.08     -15.15      -0.00
     status=first       0.86       0.16       5.39       0.00       2.36
    status=second      -0.16       0.18      -0.91       0.36       0.85
     status=third      -0.92       0.15      -6.12       0.00       0.40
        age=child       1.06       0.25       4.30       0.00       2.89
       sex=female       2.42       0.14      17.04       0.00      11.25

The next examples shows how to handle singularities in data sets
(:download:`logreg-singularities.py <code/logreg-singularities.py>`).

.. literalinclude:: code/logreg-singularities.py

The first few lines of the output of this script are::

    <=50K <=50K
    <=50K <=50K
    <=50K <=50K
    >50K >50K
    <=50K >50K

    class attribute = y
    class values = <>50K, <=50K>

                               Attribute       beta  st. error     wald Z          P OR=exp(beta)

                               Intercept       6.62      -0.00       -inf       0.00
                                     age      -0.04       0.00       -inf       0.00       0.96
                                  fnlwgt      -0.00       0.00       -inf       0.00       1.00
                           education-num      -0.28       0.00       -inf       0.00       0.76
                 marital-status=Divorced       4.29       0.00        inf       0.00      72.62
            marital-status=Never-married       3.79       0.00        inf       0.00      44.45
                marital-status=Separated       3.46       0.00        inf       0.00      31.95
                  marital-status=Widowed       3.85       0.00        inf       0.00      46.96
    marital-status=Married-spouse-absent       3.98       0.00        inf       0.00      53.63
        marital-status=Married-AF-spouse       4.01       0.00        inf       0.00      55.19
                 occupation=Tech-support      -0.32       0.00       -inf       0.00       0.72

If :obj:`remove_singular` is set to 0, inducing a logistic regression
classifier would return an error::

    Traceback (most recent call last):
      File "logreg-singularities.py", line 4, in <module>
        lr = classification.logreg.LogRegLearner(table, removeSingular=0)
      File "/home/jure/devel/orange/Orange/classification/logreg.py", line 255, in LogRegLearner
        return lr(examples, weightID)
      File "/home/jure/devel/orange/Orange/classification/logreg.py", line 291, in __call__
        lr = learner(examples, weight)
    orange.KernelException: 'orange.LogRegLearner': singularity in workclass=Never-worked

We can see that the attribute workclass is causing a singularity.

The example below shows how the use of stepwise logistic regression can help to
gain in classification performance (:download:`logreg-stepwise.py <code/logreg-stepwise.py>`):

.. literalinclude:: code/logreg-stepwise.py

The output of this script is::

    Learner      CA
    logistic     0.841
    filtered     0.846

    Number of times attributes were used in cross-validation:
     1 x a21
    10 x a22
     8 x a23
     7 x a24
     1 x a25
    10 x a26
    10 x a27
     3 x a28
     7 x a29
     9 x a31
     2 x a16
     7 x a12
     1 x a32
     8 x a15
    10 x a14
     4 x a17
     7 x a30
    10 x a11
     1 x a10
     1 x a13
    10 x a34
     2 x a19
     1 x a18
    10 x a3
    10 x a5
     4 x a4
     4 x a7
     8 x a6
    10 x a9
    10 x a8
