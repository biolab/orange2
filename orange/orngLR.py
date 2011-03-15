from Orange.classification.logreg import \
    dump as printOUT,\
    dump as printOut,\
    has_discrete_values as hasDiscreteValues,\
    LogRegLearner,\
    LogRegLearner as LogRegLearnerClass,\
    UnivariateLogRegLearner as Univariate_LogRegLearner,\
    UnivariateLogRegLearner as Univariate_LogRegLearner_Class,\
    UnivariateLogRegClassifier as UnivariateLogRegClassifier,\
    LogRegLearnerGetPriors as LogRegLearner_getPriors,\
    LogRegLearnerGetPriors as LogRegLearnerClass_getPriors,\
    LogRegLearnerGetPriorsOneTable as LogRegLearnerClass_getPriors_OneTable,\
    pr as Pr,\
    lh,\
    diag,\
    SimpleFitter as simpleFitter,\
    pr_bx as Pr_bx,\
    BayesianFitter as bayesianFitter,\
    StepWiseFSS,\
    StepWiseFSS as StepWiseFSS_class,\
    get_likelihood as getLikelihood,\
    StepWiseFSSFilter as StepWiseFSS_Filter,\
    StepWiseFSSFilter as StepWiseFSS_Filter_class,\
    lchisqprob,\
    zprob

'''
RENAMED METHODS:

BayesianFitter.createArrayData -> BayesianFitter.create_array_data
BayesianFitter.estimateBeta -> BayesianFitter.estimate_beta
'''
