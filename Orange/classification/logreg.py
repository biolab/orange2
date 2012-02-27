import Orange
from Orange.misc import deprecated_keywords, deprecated_members
import math
from numpy import dot, array, identity, reshape, diagonal, \
    transpose, concatenate, sqrt, sign
from numpy.linalg import inv
from Orange.core import LogRegClassifier, LogRegFitter, LogRegFitter_Cholesky

def dump(classifier):
    """ Return a formatted string describing the logistic regression model

    :param classifier: logistic regression classifier.
    """

    # print out class values
    out = ['']
    out.append("class attribute = " + classifier.domain.class_var.name)
    out.append("class values = " + str(classifier.domain.class_var.values))
    out.append('')
    
    # get the longest attribute name
    longest=0
    for at in classifier.continuized_domain.features:
        if len(at.name)>longest:
            longest=len(at.name)

    # print out the head
    formatstr = "%"+str(longest)+"s %10s %10s %10s %10s %10s"
    out.append(formatstr % ("Feature", "beta", "st. error", "wald Z", "P", "OR=exp(beta)"))
    out.append('')
    formatstr = "%"+str(longest)+"s %10.2f %10.2f %10.2f %10.2f"    
    out.append(formatstr % ("Intercept", classifier.beta[0], classifier.beta_se[0], classifier.wald_Z[0], classifier.P[0]))
    formatstr = "%"+str(longest)+"s %10.2f %10.2f %10.2f %10.2f %10.2f"    
    for i in range(len(classifier.continuized_domain.features)):
        out.append(formatstr % (classifier.continuized_domain.features[i].name, classifier.beta[i+1], classifier.beta_se[i+1], classifier.wald_Z[i+1], abs(classifier.P[i+1]), math.exp(classifier.beta[i+1])))

    return '\n'.join(out)
        

def has_discrete_values(domain):
    """
    Return 1 if the given domain contains any discrete features, else 0.

    :param domain: domain.
    :type domain: :class:`Orange.data.Domain`
    """
    return any(at.var_type == Orange.feature.Type.Discrete
               for at in domain.features)


class LogRegLearner(Orange.classification.Learner):
    """ Logistic regression learner.

    Returns either a learning algorithm (instance of
    :obj:`LogRegLearner`) or, if data is provided, a fitted model
    (instance of :obj:`LogRegClassifier`).

    :param data: data table; it may contain discrete and continuous features
    :type data: Orange.data.Table
    :param weight_id: the ID of the weight meta attribute
    :type weight_id: int
    :param remove_singular: automated removal of constant
        features and singularities (default: `False`)
    :type remove_singular: bool
    :param fitter: the fitting algorithm (default: :obj:`LogRegFitter_Cholesky`)
    :param stepwise_lr: enables stepwise feature selection (default: `False`)
    :type stepwise_lr: bool
    :param add_crit: threshold for adding a feature in stepwise
        selection (default: 0.2)
    :type add_crit: float
    :param delete_crit: threshold for removing a feature in stepwise
        selection (default: 0.3)
    :type delete_crit: float
    :param num_features: number of features in stepwise selection
        (default: -1, no limit)
    :type num_features: int
    :rtype: :obj:`LogRegLearner` or :obj:`LogRegClassifier`

    """

    @deprecated_keywords({"weightID": "weight_id"})
    def __new__(cls, data=None, weight_id=0, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if data:
            self.__init__(**argkw)
            return self.__call__(data, weight_id)
        else:
            return self

    @deprecated_keywords({"removeSingular": "remove_singular"})
    def __init__(self, remove_singular=0, fitter = None, **kwds):
        self.__dict__.update(kwds)
        self.remove_singular = remove_singular
        self.fitter = None

    @deprecated_keywords({"examples": "data"})
    def __call__(self, data, weight=0):
        """Fit a model to the given data.

        :param data: Data instances.
        :type data: :class:`~Orange.data.Table`
        :param weight: Id of meta attribute with instance weights
        :type weight: int
        :rtype: :class:`~Orange.classification.logreg.LogRegClassifier`
        """
        imputer = getattr(self, "imputer", None) or None
        if getattr(self, "remove_missing", 0):
            data = Orange.core.Preprocessor_dropMissing(data)
##        if hasDiscreteValues(examples.domain):
##            examples = createNoDiscTable(examples)
        if not len(data):
            return None
        if getattr(self, "stepwise_lr", 0):
            add_crit = getattr(self, "add_crit", 0.2)
            delete_crit = getattr(self, "delete_crit", 0.3)
            num_features = getattr(self, "num_features", -1)
            attributes = StepWiseFSS(data, add_crit= add_crit,
                delete_crit=delete_crit, imputer = imputer, num_features= num_features)
            tmp_domain = Orange.data.Domain(attributes,
                data.domain.class_var)
            tmp_domain.addmetas(data.domain.getmetas())
            data = data.select(tmp_domain)
        learner = Orange.core.LogRegLearner() # Yes, it has to be from core.
        learner.imputer_constructor = imputer
        if imputer:
            data = self.imputer(data)(data)
        data = Orange.core.Preprocessor_dropMissing(data)
        if self.fitter:
            learner.fitter = self.fitter
        if self.remove_singular:
            lr = learner.fit_model(data, weight)
        else:
            lr = learner(data, weight)
        while isinstance(lr, Orange.feature.Descriptor):
            if isinstance(lr.getValueFrom, Orange.core.ClassifierFromVar) and isinstance(lr.getValueFrom.transformer, Orange.core.Discrete2Continuous):
                lr = lr.getValueFrom.variable
            attributes = data.domain.features[:]
            if lr in attributes:
                attributes.remove(lr)
            else:
                attributes.remove(lr.getValueFrom.variable)
            new_domain = Orange.data.Domain(attributes, 
                data.domain.class_var)
            new_domain.addmetas(data.domain.getmetas())
            data = data.select(new_domain)
            lr = learner.fit_model(data, weight)
        return lr

LogRegLearner = deprecated_members({"removeSingular": "remove_singular",
                                    "weightID": "weight_id",
                                    "stepwiseLR": "stepwise_lr",
                                    "addCrit": "add_crit",
                                    "deleteCrit": "delete_crit",
                                    "numFeatures": "num_features",
                                    "removeMissing": "remove_missing"
                                    })(LogRegLearner)

class UnivariateLogRegLearner(Orange.classification.Learner):
    def __new__(cls, data=None, **argkw):
        self = Orange.classification.Learner.__new__(cls, **argkw)
        if data:
            self.__init__(**argkw)
            return self.__call__(data)
        else:
            return self

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    @deprecated_keywords({"examples": "data"})
    def __call__(self, data):
        data = createFullNoDiscTable(data)
        classifiers = map(lambda x: LogRegLearner(Orange.core.Preprocessor_dropMissing(
            data.select(Orange.data.Domain(x, 
                data.domain.class_var)))), data.domain.features)
        maj_classifier = LogRegLearner(Orange.core.Preprocessor_dropMissing
            (data.select(Orange.data.Domain(data.domain.class_var))))
        beta = [maj_classifier.beta[0]] + [x.beta[1] for x in classifiers]
        beta_se = [maj_classifier.beta_se[0]] + [x.beta_se[1] for x in classifiers]
        P = [maj_classifier.P[0]] + [x.P[1] for x in classifiers]
        wald_Z = [maj_classifier.wald_Z[0]] + [x.wald_Z[1] for x in classifiers]
        domain = data.domain

        return Univariate_LogRegClassifier(beta = beta, beta_se = beta_se, P = P, wald_Z = wald_Z, domain = domain)

class UnivariateLogRegClassifier(Orange.classification.Classifier):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __call__(self, instance, result_type = Orange.classification.Classifier.GetValue):
        # classification not implemented yet. For now its use is only to
        # provide regression coefficients and its statistics
        raise NotImplemented
    

class LogRegLearnerGetPriors(object):
    def __new__(cls, data=None, weight_id=0, **argkw):
        self = object.__new__(cls)
        if data:
            self.__init__(**argkw)
            return self.__call__(data, weight_id)
        else:
            return self

    @deprecated_keywords({"removeSingular": "remove_singular"})
    def __init__(self, remove_singular=0, **kwds):
        self.__dict__.update(kwds)
        self.remove_singular = remove_singular

    @deprecated_keywords({"examples": "data"})
    def __call__(self, data, weight=0):
        # next function changes data set to a extended with unknown values 
        def createLogRegExampleTable(data, weight_id):
            sets_of_data = []
            for at in data.domain.features:
                # za vsak atribut kreiraj nov newExampleTable new_data
                # v dataOrig, dataFinal in new_data dodaj nov atribut -- continuous variable
                if at.var_type == Orange.feature.Type.Continuous:
                    at_disc = Orange.feature.Continuous(at.name+ "Disc")
                    new_domain = Orange.data.Domain(data.domain.features+[at_disc,data.domain.class_var])
                    new_domain.addmetas(data.domain.getmetas())
                    new_data = Orange.data.Table(new_domain,data)
                    alt_data = Orange.data.Table(new_domain,data)
                    for i,d in enumerate(new_data):
                        d[at_disc] = 0
                        d[weight_id] = 1*data[i][weight_id]
                    for i,d in enumerate(alt_data):
                        d[at_disc] = 1
                        d[at] = 0
                        d[weight_id] = 0.000001*data[i][weight_id]
                elif at.var_type == Orange.feature.Type.Discrete:
                # v dataOrig, dataFinal in new_data atributu "at" dodaj ee  eno  vreednost, ki ima vrednost kar  ime atributa +  "X"
                    at_new = Orange.feature.Discrete(at.name, values = at.values + [at.name+"X"])
                    new_domain = Orange.data.Domain(filter(lambda x: x!=at, data.domain.features)+[at_new,data.domain.class_var])
                    new_domain.addmetas(data.domain.getmetas())
                    new_data = Orange.data.Table(new_domain,data)
                    alt_data = Orange.data.Table(new_domain,data)
                    for i,d in enumerate(new_data):
                        d[at_new] = data[i][at]
                        d[weight_id] = 1*data[i][weight_id]
                    for i,d in enumerate(alt_data):
                        d[at_new] = at.name+"X"
                        d[weight_id] = 0.000001*data[i][weight_id]
                new_data.extend(alt_data)
                sets_of_data.append(new_data)
            return sets_of_data
                  
        learner = LogRegLearner(imputer=Orange.feature.imputation.ImputerConstructor_average(),
            remove_singular = self.remove_singular)
        # get Original Model
        orig_model = learner(data, weight)
        if orig_model.fit_status:
            print "Warning: model did not converge"

        # get extended Model (you should not change data)
        if weight == 0:
            weight = Orange.feature.Descriptor.new_meta_id()
            data.addMetaAttribute(weight, 1.0)
        extended_set_of_examples = createLogRegExampleTable(data, weight)
        extended_models = [learner(extended_examples, weight) \
                           for extended_examples in extended_set_of_examples]

##        print examples[0]
##        printOUT(orig_model)
##        print orig_model.domain
##        print orig_model.beta
##        print orig_model.beta[orig_model.continuized_domain.features[-1]]
##        for i,m in enumerate(extended_models):
##            print examples.domain.features[i]
##            printOUT(m)
            
        
        # izracunas odstopanja
        # get sum of all betas
        beta = 0
        betas_ap = []
        for m in extended_models:
            beta_add = m.beta[m.continuized_domain.features[-1]]
            betas_ap.append(beta_add)
            beta = beta + beta_add
        
        # substract it from intercept
        #print "beta", beta
        logistic_prior = orig_model.beta[0]+beta
        
        # compare it to bayes prior
        bayes = Orange.classification.bayes.NaiveLearner(data)
        bayes_prior = math.log(bayes.distribution[1]/bayes.distribution[0])

        # normalize errors
##        print "bayes", bayes_prior
##        print "lr", orig_model.beta[0]
##        print "lr2", logistic_prior
##        print "dist", Orange.statistics.distribution.Distribution(examples.domain.class_var,examples)
##        print "prej", betas_ap

        # error normalization - to avoid errors due to assumption of independence of unknown values
        dif = bayes_prior - logistic_prior
        positives = sum(filter(lambda x: x>=0, betas_ap))
        negatives = -sum(filter(lambda x: x<0, betas_ap))
        if not negatives == 0:
            kPN = positives/negatives
            diffNegatives = dif/(1+kPN)
            diffPositives = kPN*diffNegatives
            kNegatives = (negatives-diffNegatives)/negatives
            kPositives = positives/(positives-diffPositives)
    ##        print kNegatives
    ##        print kPositives

            for i,b in enumerate(betas_ap):
                if b<0: betas_ap[i]*=kNegatives
                else: betas_ap[i]*=kPositives
        #print "potem", betas_ap

        # vrni originalni model in pripadajoce apriorne niclele
        return (orig_model, betas_ap)
        #return (bayes_prior,orig_model.beta[examples.domain.class_var],logistic_prior)

LogRegLearnerGetPriors = deprecated_members({"removeSingular":
                                                 "remove_singular"}
)(LogRegLearnerGetPriors)

class LogRegLearnerGetPriorsOneTable:
    @deprecated_keywords({"removeSingular": "remove_singular"})
    def __init__(self, remove_singular=0, **kwds):
        self.__dict__.update(kwds)
        self.remove_singular = remove_singular

    @deprecated_keywords({"examples": "data"})
    def __call__(self, data, weight=0):
        # next function changes data set to a extended with unknown values 
        def createLogRegExampleTable(data, weightID):
            finalData = Orange.data.Table(data)
            orig_data = Orange.data.Table(data)
            for at in data.domain.features:
                # za vsak atribut kreiraj nov newExampleTable newData
                # v dataOrig, dataFinal in newData dodaj nov atribut -- continuous variable
                if at.var_type == Orange.feature.Type.Continuous:
                    atDisc = Orange.feature.Continuous(at.name + "Disc")
                    newDomain = Orange.data.Domain(orig_data.domain.features+[atDisc,data.domain.class_var])
                    newDomain.addmetas(newData.domain.getmetas())
                    finalData = Orange.data.Table(newDomain,finalData)
                    newData = Orange.data.Table(newDomain,orig_data)
                    orig_data = Orange.data.Table(newDomain,orig_data)
                    for d in orig_data:
                        d[atDisc] = 0
                    for d in finalData:
                        d[atDisc] = 0
                    for i,d in enumerate(newData):
                        d[atDisc] = 1
                        d[at] = 0
                        d[weightID] = 100*data[i][weightID]
                        
                elif at.var_type == Orange.feature.Type.Discrete:
                # v dataOrig, dataFinal in newData atributu "at" dodaj ee  eno  vreednost, ki ima vrednost kar  ime atributa +  "X"
                    at_new = Orange.feature.Discrete(at.name, values = at.values + [at.name+"X"])
                    newDomain = Orange.data.Domain(filter(lambda x: x!=at, orig_data.domain.features)+[at_new,orig_data.domain.class_var])
                    newDomain.addmetas(orig_data.domain.getmetas())
                    temp_finalData = Orange.data.Table(finalData)
                    finalData = Orange.data.Table(newDomain,finalData)
                    newData = Orange.data.Table(newDomain,orig_data)
                    temp_origData = Orange.data.Table(orig_data)
                    orig_data = Orange.data.Table(newDomain,orig_data)
                    for i,d in enumerate(orig_data):
                        d[at_new] = temp_origData[i][at]
                    for i,d in enumerate(finalData):
                        d[at_new] = temp_finalData[i][at]
                    for i,d in enumerate(newData):
                        d[at_new] = at.name+"X"
                        d[weightID] = 10*data[i][weightID]
                finalData.extend(newData)
            return finalData
                  
        learner = LogRegLearner(imputer = Orange.feature.imputation.ImputerConstructor_average(), removeSingular = self.remove_singular)
        # get Original Model
        orig_model = learner(data,weight)

        # get extended Model (you should not change data)
        if weight == 0:
            weight = Orange.feature.Descriptor.new_meta_id()
            data.addMetaAttribute(weight, 1.0)
        extended_examples = createLogRegExampleTable(data, weight)
        extended_model = learner(extended_examples, weight)

##        print examples[0]
##        printOUT(orig_model)
##        print orig_model.domain
##        print orig_model.beta

##        printOUT(extended_model)        
        # izracunas odstopanja
        # get sum of all betas
        beta = 0
        betas_ap = []
        for m in extended_models:
            beta_add = m.beta[m.continuized_domain.features[-1]]
            betas_ap.append(beta_add)
            beta = beta + beta_add
        
        # substract it from intercept
        #print "beta", beta
        logistic_prior = orig_model.beta[0]+beta
        
        # compare it to bayes prior
        bayes = Orange.classification.bayes.NaiveLearner(data)
        bayes_prior = math.log(bayes.distribution[1]/bayes.distribution[0])

        # normalize errors
        #print "bayes", bayes_prior
        #print "lr", orig_model.beta[0]
        #print "lr2", logistic_prior
        #print "dist", Orange.statistics.distribution.Distribution(examples.domain.class_var,examples)
        k = (bayes_prior-orig_model.beta[0])/(logistic_prior-orig_model.beta[0])
        #print "prej", betas_ap
        betas_ap = [k*x for x in betas_ap]                
        #print "potem", betas_ap

        # vrni originalni model in pripadajoce apriorne niclele
        return (orig_model, betas_ap)
        #return (bayes_prior,orig_model.beta[data.domain.class_var],logistic_prior)

LogRegLearnerGetPriorsOneTable = deprecated_members({"removeSingular":
                                                         "remove_singular"}
)(LogRegLearnerGetPriorsOneTable)


######################################
#### Fitters for logistic regression (logreg) learner ####
######################################

def pr(x, betas):
    k = math.exp(dot(x, betas))
    return k / (1+k)

def lh(x,y,betas):
    llh = 0.0
    for i,x_i in enumerate(x):
        pr = pr(x_i,betas)
        llh += y[i]*math.log(max(pr,1e-6)) + (1-y[i])*log(max(1-pr,1e-6))
    return llh


def diag(vector):
    mat = identity(len(vector))
    for i,v in enumerate(vector):
        mat[i][i] = v
    return mat
    
class SimpleFitter(LogRegFitter):
    def __init__(self, penalty=0, se_penalty = False):
        self.penalty = penalty
        self.se_penalty = se_penalty

    def __call__(self, data, weight=0):
        ml = data.native(0)
        for i in range(len(data.domain.features)):
          a = data.domain.features[i]
          if a.var_type == Orange.feature.Type.Discrete:
            for m in ml:
              m[i] = a.values.index(m[i])
        for m in ml:
          m[-1] = data.domain.class_var.values.index(m[-1])
        Xtmp = array(ml)
        y = Xtmp[:,-1]   # true probabilities (1's or 0's)
        one = reshape(array([1]*len(data)), (len(data),1)) # intercept column
        X=concatenate((one, Xtmp[:,:-1]),1)  # intercept first, then data

        betas = array([0.0] * (len(data.domain.features)+1))
        oldBetas = array([1.0] * (len(data.domain.features)+1))
        N = len(data)

        pen_matrix = array([self.penalty] * (len(data.domain.features)+1))
        if self.se_penalty:
            p = array([pr(X[i], betas) for i in range(len(data))])
            W = identity(len(data))
            pp = p * (1.0-p)
            for i in range(N):
                W[i,i] = pp[i]
            se = sqrt(diagonal(inv(dot(transpose(X), dot(W, X)))))
            for i,p in enumerate(pen_matrix):
                pen_matrix[i] *= se[i]
        # predict the probability for an instance, x and betas are vectors
        # start the computation
        likelihood = 0.
        likelihood_new = 1.
        while abs(likelihood - likelihood_new)>1e-5:
            likelihood = likelihood_new
            oldBetas = betas
            p = array([pr(X[i], betas) for i in range(len(data))])

            W = identity(len(data))
            pp = p * (1.0-p)
            for i in range(N):
                W[i,i] = pp[i]

            WI = inv(W)
            z = dot(X, betas) + dot(WI, y - p)

            tmpA = inv(dot(transpose(X), dot(W, X))+diag(pen_matrix))
            tmpB = dot(transpose(X), y-p)
            betas = oldBetas + dot(tmpA,tmpB)
#            betaTemp = dot(dot(dot(dot(tmpA,transpose(X)),W),X),oldBetas)
#            print betaTemp
#            tmpB = dot(transpose(X), dot(W, z))
#            betas = dot(tmpA, tmpB)
            likelihood_new = lh(X,y,betas)-self.penalty*sum([b*b for b in betas])
            print likelihood_new

            
            
##        XX = sqrt(diagonal(inv(dot(transpose(X),X))))
##        yhat = array([pr(X[i], betas) for i in range(len(data))])
##        ss = sum((y - yhat) ** 2) / (N - len(data.domain.features) - 1)
##        sigma = math.sqrt(ss)
        p = array([pr(X[i], betas) for i in range(len(data))])
        W = identity(len(data))
        pp = p * (1.0-p)
        for i in range(N):
            W[i,i] = pp[i]
        diXWX = sqrt(diagonal(inv(dot(transpose(X), dot(W, X)))))
        xTemp = dot(dot(inv(dot(transpose(X), dot(W, X))),transpose(X)),y)
        beta = []
        beta_se = []
        print "likelihood ridge", likelihood
        for i in range(len(betas)):
            beta.append(betas[i])
            beta_se.append(diXWX[i])
        return (self.OK, beta, beta_se, 0)

def pr_bx(bx):
    if bx > 35:
        return 1
    if bx < -35:
        return 0
    return exp(bx)/(1+exp(bx))

class BayesianFitter(LogRegFitter):
    def __init__(self, penalty=0, anch_examples=[], tau = 0):
        self.penalty = penalty
        self.anch_examples = anch_examples
        self.tau = tau

    def create_array_data(self,data):
        if not len(data):
            return (array([]),array([]))
        # convert data to numeric
        ml = data.native(0)
        for i,a in enumerate(data.domain.features):
          if a.var_type == Orange.feature.Type.Discrete:
            for m in ml:
              m[i] = a.values.index(m[i])
        for m in ml:
          m[-1] = data.domain.class_var.values.index(m[-1])
        Xtmp = array(ml)
        y = Xtmp[:,-1]   # true probabilities (1's or 0's)
        one = reshape(array([1]*len(data)), (len(data),1)) # intercept column
        X=concatenate((one, Xtmp[:,:-1]),1)  # intercept first, then data
        return (X,y)
    
    def __call__(self, data, weight=0):
        (X,y)=self.create_array_data(data)

        exTable = Orange.data.Table(data.domain)
        for id,ex in self.anch_examples:
            exTable.extend(Orange.data.Table(ex,data.domain))
        (X_anch,y_anch)=self.create_array_data(exTable)

        betas = array([0.0] * (len(data.domain.features)+1))

        likelihood,betas = self.estimate_beta(X,y,betas,[0]*(len(betas)),X_anch,y_anch)

        # get attribute groups atGroup = [(startIndex, number of values), ...)
        ats = data.domain.features
        atVec=reduce(lambda x,y: x+[(y,not y==x[-1][0])], [a.getValueFrom and a.getValueFrom.whichVar or a for a in ats],[(ats[0].getValueFrom and ats[0].getValueFrom.whichVar or ats[0],0)])[1:]
        atGroup=[[0,0]]
        for v_i,v in enumerate(atVec):
            if v[1]==0: atGroup[-1][1]+=1
            else:       atGroup.append([v_i,1])
        
        # compute zero values for attributes
        sumB = 0.
        for ag in atGroup:
            X_temp = concatenate((X[:,:ag[0]+1],X[:,ag[0]+1+ag[1]:]),1)
            if X_anch:
                X_anch_temp = concatenate((X_anch[:,:ag[0]+1],X_anch[:,ag[0]+1+ag[1]:]),1)
            else: X_anch_temp = X_anch
##            print "1", concatenate((betas[:i+1],betas[i+2:]))
##            print "2", betas
            likelihood_temp,betas_temp=self.estimate_beta(X_temp,y,concatenate((betas[:ag[0]+1],betas[ag[0]+ag[1]+1:])),[0]+[1]*(len(betas)-1-ag[1]),X_anch_temp,y_anch)
            print "finBetas", betas, betas_temp
            print "betas", betas[0], betas_temp[0]
            sumB += betas[0]-betas_temp[0]
        apriori = Orange.statistics.distribution.Distribution(data.domain.class_var, data)
        aprioriProb = apriori[0]/apriori.abs
        
        print "koncni rezultat", sumB, math.log((1-aprioriProb)/aprioriProb), betas[0]
            
        beta = []
        beta_se = []
        print "likelihood2", likelihood
        for i in range(len(betas)):
            beta.append(betas[i])
            beta_se.append(0.0)
        return (self.OK, beta, beta_se, 0)

     
        
    def estimate_beta(self,X,y,betas,const_betas,X_anch,y_anch):
        N,N_anch = len(y),len(y_anch)
        r,r_anch = array([dot(X[i], betas) for i in range(N)]),\
                   array([dot(X_anch[i], betas) for i in range(N_anch)])
        p    = array([pr_bx(ri) for ri in r])
        X_sq = X*X

        max_delta      = [1.]*len(const_betas)
        likelihood     = -1.e+10
        likelihood_new = -1.e+9
        while abs(likelihood - likelihood_new)>0.01 and max(max_delta)>0.01:
            likelihood = likelihood_new
            print likelihood
            betas_temp = [b for b in betas]
            for j in range(len(betas)):
                if const_betas[j]: continue
                dl = dot(X[:,j], transpose(y-p))
                for xi,x in enumerate(X_anch):
                    dl += self.penalty*x[j]*(y_anch[xi] - pr_bx(r_anch[xi]*self.penalty))

                ddl = dot(X_sq[:,j], transpose(p*(1-p)))
                for xi,x in enumerate(X_anch):
                    ddl += self.penalty*x[j]*pr_bx(r[xi]*self.penalty)*(1-pr_bx(r[xi]*self.penalty))

                if j==0:
                    dv = dl/max(ddl,1e-6)
                elif betas[j] == 0: # special handling due to non-defined first and second derivatives
                    dv = (dl-self.tau)/max(ddl,1e-6)
                    if dv < 0:
                        dv = (dl+self.tau)/max(ddl,1e-6)
                        if dv > 0:
                            dv = 0
                else:
                    dl -= sign(betas[j])*self.tau
                    dv = dl/max(ddl,1e-6)
                    if not sign(betas[j] + dv) == sign(betas[j]):
                        dv = -betas[j]
                dv = min(max(dv,-max_delta[j]),max_delta[j])
                r+= X[:,j]*dv
                p = array([pr_bx(ri) for ri in r])
                if N_anch:
                    r_anch+=X_anch[:,j]*dv
                betas[j] += dv
                max_delta[j] = max(2*abs(dv),max_delta[j]/2)
            likelihood_new = lh(X,y,betas)
            for xi,x in enumerate(X_anch):
                try:
                    likelihood_new += y_anch[xi]*r_anch[xi]*self.penalty-log(1+exp(r_anch[xi]*self.penalty))
                except:
                    likelihood_new += r_anch[xi]*self.penalty*(y_anch[xi]-1)
            likelihood_new -= sum([abs(b) for b in betas[1:]])*self.tau
            if likelihood_new < likelihood:
                max_delta = [md/4 for md in max_delta]
                likelihood_new = likelihood
                likelihood = likelihood_new + 1.
                betas = [b for b in betas_temp]
        print "betas", betas
        print "init_like", likelihood_new
        print "pure_like", lh(X,y,betas)
        return (likelihood,betas)
    
############################################################
#  Feature subset selection for logistic regression

@deprecated_keywords({"examples": "data"})
def get_likelihood(fitter, data):
    res = fitter(data)
    if res[0] in [fitter.OK]: #, fitter.Infinity, fitter.Divergence]:
       status, beta, beta_se, likelihood = res
       if sum([abs(b) for b in beta])<sum([abs(b) for b in beta_se]):
           return -100*len(data)
       return likelihood
    else:
       return -100*len(data)
        


class StepWiseFSS(object):
  """
  A learning algorithm for logistic regression that implements a
  stepwise feature subset selection as described in Applied Logistic
  Regression (Hosmer and Lemeshow, 2000).

  Each step of the algorithm is composed of two parts. The first is
  backward elimination in which the least significant variable in the
  model is removed if its p-value is above the prescribed threshold
  :obj:`delete_crit`. The second step is forward selection in which
  all variables are tested for addition to the model, and the one with
  the most significant contribution is added if the corresponding
  p-value is smaller than the prescribed :obj:d`add_crit`. The
  algorithm stops when no more variables can be added or removed.

  The model can be additionaly constrained by setting
  :obj:`num_features` to a non-negative value. The algorithm will then
  stop when the number of variables exceeds the given limit.

  Significances are assesed by the likelihood ratio chi-square
  test. Normal F test is not appropriate since the errors are assumed
  to follow a binomial distribution.

  The class constructor returns an instance of learning algorithm or,
  if given training data, a list of selected variables.

  :param table: training data.
  :type table: Orange.data.Table

  :param add_crit: threshold for adding a variable (default: 0.2)
  :type add_crit: float

  :param delete_crit: threshold for removing a variable
      (default: 0.3); should be higher than :obj:`add_crit`.
  :type delete_crit: float

  :param num_features: maximum number of selected features,
      use -1 for infinity.
  :type num_features: int
  :rtype: :obj:`StepWiseFSS` or list of features

  """

  def __new__(cls, data=None, **argkw):
      self = object.__new__(cls)
      if data is not None:
          self.__init__(**argkw)
          return self.__call__(data)
      else:
          return self

  @deprecated_keywords({"addCrit": "add_crit", "deleteCrit": "delete_crit",
                        "numFeatures": "num_features"})
  def __init__(self, add_crit=0.2, delete_crit=0.3, num_features=-1, **kwds):
    self.__dict__.update(kwds)
    self.add_crit = add_crit
    self.delete_crit = delete_crit
    self.num_features = num_features

  def __call__(self, examples):
    if getattr(self, "imputer", 0):
        examples = self.imputer(examples)(examples)
    if getattr(self, "removeMissing", 0):
        examples = Orange.core.Preprocessor_dropMissing(examples)
    continuizer = Orange.preprocess.DomainContinuizer(zeroBased=1,
        continuousTreatment=Orange.preprocess.DomainContinuizer.Leave,
                                           multinomialTreatment = Orange.preprocess.DomainContinuizer.FrequentIsBase,
                                           classTreatment = Orange.preprocess.DomainContinuizer.Ignore)
    attr = []
    remain_attr = examples.domain.features[:]

    # get LL for Majority Learner 
    tempDomain = Orange.data.Domain(attr,examples.domain.class_var)
    #tempData  = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))
    tempData  = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))

    ll_Old = get_likelihood(LogRegFitter_Cholesky(), tempData)
    ll_Best = -1000000
    length_Old = float(len(tempData))

    stop = 0
    while not stop:
        # LOOP until all variables are added or no further deletion nor addition of attribute is possible
        worstAt = None
        # if there are more than 1 attribute then perform backward elimination
        if len(attr) >= 2:
            minG = 1000
            worstAt = attr[0]
            ll_Best = ll_Old
            length_Best = length_Old
            for at in attr:
                # check all attribute whether its presence enough increases LL?

                tempAttr = filter(lambda x: x!=at, attr)
                tempDomain = Orange.data.Domain(tempAttr,examples.domain.class_var)
                tempDomain.addmetas(examples.domain.getmetas())
                # domain, calculate P for LL improvement.
                tempDomain  = continuizer(Orange.core.Preprocessor_dropMissing(examples.select(tempDomain)))
                tempData = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))

                ll_Delete = get_likelihood(LogRegFitter_Cholesky(), tempData)
                length_Delete = float(len(tempData))
                length_Avg = (length_Delete + length_Old)/2.0

                G=-2*length_Avg*(ll_Delete/length_Delete-ll_Old/length_Old)

                # set new worst attribute
                if G<minG:
                    worstAt = at
                    minG=G
                    ll_Best = ll_Delete
                    length_Best = length_Delete
            # deletion of attribute

            if worstAt.var_type==Orange.feature.Type.Continuous:
                P=lchisqprob(minG,1);
            else:
                P=lchisqprob(minG,len(worstAt.values)-1);
            if P>=self.delete_crit:
                attr.remove(worstAt)
                remain_attr.append(worstAt)
                nodeletion=0
                ll_Old = ll_Best
                length_Old = length_Best
            else:
                nodeletion=1
        else:
            nodeletion = 1
            # END OF DELETION PART

        # if enough attributes has been chosen, stop the procedure
        if self.num_features>-1 and len(attr)>=self.num_features:
            remain_attr=[]

        # for each attribute in the remaining
        maxG=-1
        ll_Best = ll_Old
        length_Best = length_Old
        bestAt = None
        for at in remain_attr:
            tempAttr = attr + [at]
            tempDomain = Orange.data.Domain(tempAttr,examples.domain.class_var)
            tempDomain.addmetas(examples.domain.getmetas())
            # domain, calculate P for LL improvement.
            tempDomain  = continuizer(Orange.core.Preprocessor_dropMissing(examples.select(tempDomain)))
            tempData = Orange.core.Preprocessor_dropMissing(examples.select(tempDomain))
            ll_New = get_likelihood(LogRegFitter_Cholesky(), tempData)

            length_New = float(len(tempData)) # get number of examples in tempData to normalize likelihood

            # P=PR(CHI^2>G), G=-2(L(0)-L(1))=2(E(0)-E(1))
            length_avg = (length_New + length_Old)/2
            G=-2*length_avg*(ll_Old/length_Old-ll_New/length_New);
            if G>maxG:
                bestAt = at
                maxG=G
                ll_Best = ll_New
                length_Best = length_New
        if not bestAt:
            stop = 1
            continue

        if bestAt.var_type==Orange.feature.Type.Continuous:
            P=lchisqprob(maxG,1);
        else:
            P=lchisqprob(maxG,len(bestAt.values)-1);
        # Add attribute with smallest P to attributes(attr)
        if P<=self.add_crit:
            attr.append(bestAt)
            remain_attr.remove(bestAt)
            ll_Old = ll_Best
            length_Old = length_Best

        if (P>self.add_crit and nodeletion) or (bestAt == worstAt):
            stop = 1

    return attr

StepWiseFSS = deprecated_members({"addCrit": "add_crit",
                                   "deleteCrit": "delete_crit",
                                   "numFeatures": "num_features"})(StepWiseFSS)


class StepWiseFSSFilter(object):
    def __new__(cls, data=None, **argkw):
        self = object.__new__(cls)
        if data:
            self.__init__(**argkw)
            return self.__call__(data)
        else:
            return self

    @deprecated_keywords({"addCrit": "add_crit", "deleteCrit": "delete_crit",
                          "numFeatures": "num_features"})
    def __init__(self, add_crit=0.2, delete_crit=0.3, num_features = -1):
        self.add_crit = add_crit
        self.delete_crit = delete_crit
        self.num_features = num_features

    @deprecated_keywords({"examples": "data"})
    def __call__(self, data):
        attr = StepWiseFSS(data, add_crit=self.add_crit,
            delete_crit= self.delete_crit, num_features= self.num_features)
        return data.select(Orange.data.Domain(attr, data.domain.class_var))

StepWiseFSSFilter = deprecated_members({"addCrit": "add_crit",
                                        "deleteCrit": "delete_crit",
                                        "numFeatures": "num_features"})\
    (StepWiseFSSFilter)


####################################
##  PROBABILITY CALCULATIONS

def lchisqprob(chisq,df):
    """
    Return the (1-tailed) probability value associated with the provided
    chi-square value and df.  Adapted from chisq.c in Gary Perlman's |Stat.
    """
    BIG = 20.0
    def ex(x):
    	BIG = 20.0
    	if x < -BIG:
    	    return 0.0
    	else:
    	    return math.exp(x)
    if chisq <=0 or df < 1:
    	return 1.0
    a = 0.5 * chisq
    if df%2 == 0:
    	even = 1
    else:
    	even = 0
    if df > 1:
    	y = ex(-a)
    if even:
    	s = y
    else:
        s = 2.0 * zprob(-math.sqrt(chisq))
    if (df > 2):
        chisq = 0.5 * (df - 1.0)
        if even:
            z = 1.0
        else:
            z = 0.5
        if a > BIG:
            if even:
            	e = 0.0
            else:
            	e = math.log(math.sqrt(math.pi))
            c = math.log(a)
            while (z <= chisq):
            	e = math.log(z) + e
            	s = s + ex(c*z-a-e)
            	z = z + 1.0
            return s
        else:
            if even:
                e = 1.0
            else:
                e = 1.0 / math.sqrt(math.pi) / math.sqrt(a)
            c = 0.0
            while (z <= chisq):
                e = e * (a/float(z))
                c = c + e
                z = z + 1.0
            return (c*y+s)
    else:
        return s


def zprob(z):
    """
    Returns the area under the normal curve 'to the left of' the given z value.
    Thus:: 

    for z<0, zprob(z) = 1-tail probability
    for z>0, 1.0-zprob(z) = 1-tail probability
    for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability

    Adapted from z.c in Gary Perlman's |Stat.
    """
    Z_MAX = 6.0    # maximum meaningful z-value
    if z == 0.0:
	x = 0.0
    else:
	y = 0.5 * math.fabs(z)
	if y >= (Z_MAX*0.5):
	    x = 1.0
	elif (y < 1.0):
	    w = y*y
	    x = ((((((((0.000124818987 * w
			-0.001075204047) * w +0.005198775019) * w
		      -0.019198292004) * w +0.059054035642) * w
		    -0.151968751364) * w +0.319152932694) * w
		  -0.531923007300) * w +0.797884560593) * y * 2.0
	else:
	    y = y - 2.0
	    x = (((((((((((((-0.000045255659 * y
			     +0.000152529290) * y -0.000019538132) * y
			   -0.000676904986) * y +0.001390604284) * y
			 -0.000794620820) * y -0.002034254874) * y
		       +0.006549791214) * y -0.010557625006) * y
		     +0.011630447319) * y -0.009279453341) * y
		   +0.005353579108) * y -0.002141268741) * y
		 +0.000535310849) * y +0.999936657524
    if z > 0.0:
	prob = ((x+1.0)*0.5)
    else:
	prob = ((1.0-x)*0.5)
    return prob

   
