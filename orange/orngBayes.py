import orange

def BayesLearner(examples = None, weightID = 0, **argkw):
    tl = apply(BayesLearnerClass, (), argkw)
    if examples:
        tl = tl(examples, weightID)
    return tl

class BayesLearnerClass:
    def __init__(self, **argkw):
        self.learner = None
        self.__dict__.update(argkw)

    def __setattr__(self, name, value):
        if name in ["m", "estimatorConstructor", "conditionalEstimatorConstructor", "conditionalEstimatorConstructorContinuous"]:
            self.learner = None
        self.__dict__[name] = value

    def __call__(self, examples, weight=0):
        if not self.learner:
            self.learner = self.createInstance()
        return self.learner(examples, weight)

    def createInstance(self):
        bayes = orange.BayesLearner()
        if hasattr(self, "estimatorConstructor"):
            bayes.estimatorConstructor = self.estimatorConstructor
            if hasattr(self, "m"):
                if hasattr(bayes.estimatorConstructor, "m"):
                    raise AttributeError, "invalid combination of attributes: 'estimatorConstructor' does not expect 'm'"
                else:
                    self.estimatorConstructor.m = self.m
        elif hasattr(self, "m"):
            bayes.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m = self.m)

        if hasattr(self, "conditionalEstimatorConstructor"):
            bayes.conditionalEstimatorConstructor = self.conditionalEstimatorConstructor
        else:
            bayes.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows()
            bayes.conditionalEstimatorConstructor.estimatorConstructor=bayes.estimatorConstructor
            
        if hasattr(self, "conditionalEstimatorConstructorContinuous"):
            bayes.conditionalEstimatorConstructorContinuous = self.conditionalEstimatorConstructorContinuous
            
        return bayes
            

def printModel(bayesclassifier):
    nValues=len(bayesclassifier.classVar.values)
    frmtStr=' %10.3f'*nValues
    classes=" "*20+ ((' %10s'*nValues) % tuple([i[:10] for i in bayesclassifier.classVar.values]))
    print classes
    print "class probabilities "+(frmtStr % tuple(bayesclassifier.probabilities.classes))
    print

    for i in bayesclassifier.probabilities:
        print "Attribute", i.variable.name
        print classes
        for v in range(len(i.variable.values)):
            print ("%20s" % i.variable.values[v][:20]) + (frmtStr % tuple(i[v].distribution))
        print

    
def toXMLCharset(s):
    import re
    p = re.compile('(<)')
    s = p.sub('&lt;', s)

    p = re.compile('(>)')
    s = p.sub('&gt;', s)
    return s

def saveXML(file, bayesclassifier, includeClasses=0):
    import types
    fopened=0
    if (type(file)==types.StringType):
        f=open(file, "wt")
        fopened=1

    f.write('<?xml version="1.0" ?>\n')
    f.write('<model name="Confined">\n')
    f.write('<description>Based on preoperative predictors computes the probability that prostate cancer is organ-confined.</description>\n')
    f.write('<author>TBA</author>\n')
    f.write('<date>1999-03-20</date>\n')
    f.write('<outcome>Organ-Confined Prostate Cancer</outcome>\n')

    if includeClasses or (len(bayesclassifier.classVar.values)>2):
        f.write('<classes>'+reduce(lambda x,y: x+"; "+y, [i for i in bayesclassifier.classVar.values])+'</classes>')
        
    f.write('<variables>\n')
    for i in bayesclassifier.domain.attributes:
        f.write('  <var>\n')
        s = '    <name>%s</name>' % i.name
        f.write(s)
        f.write('    <type>categorical</type>\n')
        f.write('    <input>pulldown</input>\n')
        f.write('    <values>')
        n = 0
        for j in i.values:
          if n: f.write(';'),
          n=1
          s = '%s' % j;
          f.write(toXMLCharset(s))
        f.write('</values>\n')
        f.write('    <default>NA</default>\n')
        f.write('    <page>1</page>\n')
        f.write('  </var>\n')
    f.write('</variables>\n')

    f.write('<pages>\n')
    f.write('  <page id="1">Patient Data</page>\n')
    f.write('</pages>\n')
    
    f.write('<modeldefinition type="naivebayes">\n')
    # prob yes, no (should be reversed)
    frmtStr=('%5.3f; '*len(bayesclassifier.classVar.values))[:-2]
    s = ('  <classprobabilities>'+frmtStr+'</classprobabilities>') % tuple(bayesclassifier.probabilities.classes.distribution)
    f.write(s)
    f.write('  <contingencymatrix>\n')
    for i in bayesclassifier.probabilities:
        # prob no, yes (reversed as above)
        s = '    <conditionalprobability attribute="%s">' % i.variable.name
        f.write(s)
        n = 0
        for j in i:
            if n: f.write(' ; '),
            n=1
            s = '%5.3f,%5.3f' % (j[0], j[1])
            f.write(frmtStr % tuple(j.distribution))
        f.write('</conditionalprobability>\n')
    f.write('  </contingencymatrix>\n')
    f.write('</modeldefinition>\n')

    f.write('</model>\n')

    if fopened:    
        f.close()
