import orange, orngMisc
reload(orngMisc)

class ImputePreprocessor:
    def __init__(self, imputeFunction):
        self.imputeFunction=imputeFunction

    def __call__(self, tab):
        self.imputeFunction(tab)
        imputed=orange.ExampleTable(tab.domain)
        for ex in tab:
            imputed.append(self.imputeFunction(ex))
        return imputed


class ImputeClassifier:
    def __init__(self, imputeFunction, classifier, weight):
        self.imputeFunction=imputeFunction
        self.classifier=classifier
        self.weight=weight

    def __call__(self, example, returnWhat):
        import pywin.debugger
#        pywin.debugger.brk()

        examples=self.imputeFunction(example)

        if example.domain.classVar.varType==orange.VarTypes.Discrete:
            probs=[0]*len(example.domain.classVar.values)
            for ex in examples:
                weight=int(ex.getmeta(self.weight))
                prob0=self.classifier(ex, orange.GetProbabilities)
                for i in range(len(prob0)):
                    probs[i] += weight*prob0[i]

            if returnWhat==orange.GetProbabilities:
                return probs
            
            prediction=orange.Value(ex.domain.classVar, orngMisc.selectBestIndex(probs))
            
            if returnWhat==orange.Value:
                return prediction
            else:
                return prediction, probs

        else:
            if returnWhat!=orange.Value:
                raise AttributeError, "cannot return probabilities for continuous classes"

            avg=N=0.0
            for ex in examples:
                pred0=self.classifies(ex)
                if not pred0.isSpecial():
                        avg+=float(pred0)
                        N += 1
            if not N:
                return orange.Value(ex.domain.classVar)
            else:
                return orange.Value(ex.domain.classVar, avg/N)


class ImputeLearner:
    def __init__(self, imputeFunction, learner, weight, **keyw):
        self.__dict__ =keyw
        self.learner=learner
        self.weight=weight
        self.imputeFunction=imputeFunction

    def __call__(self, tab):
        tabi=ImputePreprocessor(self.imputeFunction)(tab)
        classifier=self.learner(tabi)
        return ImputeClassifier(self.imputeFunction, classifier, self.weight)




class Imputator:
    def learn(self, par):
        pass
    
    def __call__(self, par):
        if orange.type(par)==orange.types.ExampleTableType:
            self.learn(par)
            return par
        else:
            return self.impute(par)
    


class ImputeRandom(Imputator):
    def impute(self, ex):
        ex=ex.domain(ex)
        for i in range(len(ex.domain)):
            if ex[i].isSpecial():
                ex[i]=ex.domain[i].randomvalue()
        return [ex]


class ImputeDefaults(Imputator):
    def __init__(self):
        self.defaults={}
        
    def impute(self, ex):
        ex=ex.domain(ex)
        for i in range(len(ex.domain)):
            if ex[i].isSpecial():
                ex[i]=self.defaults.get(ex.domain[i].name, ex[i])
        return [ex]

    def setDefaults(self, **defaultPairs):
        self.defaults.update(defaultPairs)

    
class ImputeMajority(ImputeDefaults):
    def learn(self, tab):
        for dist in orange.DomainDistributions(tab):
            self.defaults[dist.variable.name]=orngMisc.selectBestIndex(dist)

class ImputeLowest(Imputator):
    def impute(self, ex):
        ex=ex.domain(ex)
        for i in range(len(ex.domain)):
            if ex[i].isSpecial():
                ex[i]=orange.Value(ex.domain[i], 0)
        return [ex]

class ImputeHighest(Imputator):
    def impute(self, ex):
        ex=ex.domain(ex)
        for i in range(len(ex.domain)):
            if ex[i].isSpecial():
                ex[i]=orange.Value(ex.domain[i], ex.domain[i].values[-1])
        return [ex]


### Test imputation on Bayes

if __name__ == '__main__':
    import os
    os.chdir("D:\\AI\\Papers\\PhD\\Podatki\\r_breast")
    tab=orange.ExampleTable(orange.TabDelimExampleGenerator("breast.tab"))


    ### introduce missing values and impute them using random imputation

    pp_missing=orange.Preprocessor_addMissing(proportions={'a1':0.7, 'a3':0.2, 'a4':0.1})
    tab_missing=pp_missing(tab)

    impute_random=ImputePreprocessor(ImputeRandom())
    tab_imputed=impute_random(tab_missing)

    for i in range(10):
        print tab[i], tab_missing[i], tab_imputed[i]


    import orngEval
    reload(orngEval)

    weightID=orange.newmetaid()
    tab.addMetaAttribute(weightID)

    pp_missing=orange.Preprocessor_missing(probabilities={'a':0.1, 'b':0.4, 'f':0.0}, randseed=1235)

    bayes=orange.BayesLearner(shortDescription="*bayes")

    impute_random=ImputeRandom()
    bayes_random_imputation=ImputeLearner(impute_random, orange.BayesLearner(), weightID)
    bayes_random_imputation.shortDescription="*bayes with random imputation"

    impute_majority=ImputeRandom()
    bayes_majority_imputation=ImputeLearner(impute_random, orange.BayesLearner(), weightID)
    bayes_majority_imputation.shortDescription="*bayes with majority imputation"

    learners=[bayes, bayes_random_imputation, bayes_majority_imputation]
    proportions=orange.frange(0.1)

    res=orngEval.LearningCurve(learners, tab, None, None, proportions, [("L", pp_missing), ("L", impute_random), ("L", impute_majority)])
    learners=orngEval.plotLearningCurveLearners("d:\\t.plt", res, proportions, learners)
