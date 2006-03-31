import orange, time

class BayesLearner(orange.Learner):
    def __new__(cls, examples = None, weightID = 0, **argkw):
        self = orange.Learner.__new__(cls, **argkw)
        if examples:
            self.__init__(**argkw)
            return self.__call__(examples, weightID)
        else:
            return self
        
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
        elif bayes.estimatorConstructor:
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
    print "class probabilities "+(frmtStr % tuple(bayesclassifier.distribution))
    print

    for i in bayesclassifier.conditionalDistributions:
        print "Attribute", i.variable.name
        print classes
        for v in range(len(i.variable.values)):
            print ("%20s" % i.variable.values[v][:20]) + (frmtStr % tuple(i[v]))
        print

    

def XC(s):
    return s.replace("<", "&lt;").replace(">", "&gt;")

# this filters the ID string, it should not contain any special symbols,
# only letters and numbers and _. We are using IDs in expressions!

def XID(s):
    import re
    p = re.compile('[^a-zA-Z0-9_]')
    return p.sub('', s)

def saveXML(file, model):
    import types
    fopened=0
    if (type(file)==types.StringType):
        f=open(file, "wt")
        fopened=1

    f.write('<?xml version="1.0" ?>\n')
    f.write('<dsscheme xmlns="http://tempuri.org/ModelDescSchema.xsd" xmlns:xs="http://www.w3.org/2001/XMLSchema-instance">\n')
#    f.write('<dsscheme>\n')
    f.write('  <name>%s</name>\n' % getattr(model, 'xName', XC(model.classVar.name)))
    if getattr(model,'xAuthor', None): f.write('<author>%s</author>\n' % model.xAuthor)
    f.write('  <date>%s</date>\n' % getattr(model, 'xDate', time.strftime('%b %d, %Y',time.localtime(time.time()))))
    if getattr(model, 'xDescription', None): f.write('<description>%s</description>\n' % model.xDescription)
    f.write('  <variables>\n')

    # here we assume that all variables are either discrete or have been categorized
    # for the later, getValueFrom has to be set by categorizer
    needTransformation = 0
    for a in model.domain.attributes:
        f.write('    <variable>\n')
        f.write('      <id>%s</id>\n' % XID(a.name))
        if a.getValueFrom: # discretized
            f.write('      <name>%s</name>\n' % getattr(a, 'xName', a.getValueFrom.whichVar.name))
            f.write('      <description>%s</description>\n' % getattr(a, 'xDescription', 'lala'))
            if getattr(a, 'xInputAsNumber', 0):
                needTransformation = 1
                f.write('      <type>numerical</type>\n')
                f.write('      <inputtype>numerical</inputtype>\n')
            else:
                f.write('      <type>categorized</type>\n')
                f.write('      <cutoffs>%s</cutoffs>\n' % reduce(lambda x,y: x+';'+y, ["%3.1f" % i for i in a.getValueFrom.transformer.points]))
                f.write('      <inputtype>%s</inputtype>\n' % getattr(a, 'xInputType', 'pulldown'))
        else:
            f.write('      <name>%s</name>\n' % getattr(a, 'xName', a.name))
            f.write('      <description>%s</description>\n' % getattr(a, 'xDescription', ''))
            f.write('      <type>categorical</type>\n')
            f.write('      <values>%s</values>\n' % reduce(lambda x,y: x+';'+y, [i for i in a.values]))
#        if getattr(a, 'xDescription', None): f.write('     <description>%s</description>\n' % i.xDescription)
            f.write('      <inputtype>%s</inputtype>\n' % getattr(a, 'xInputType', 'pulldown'))
        if getattr(a, 'xDefault', None): f.write('      <default>%s</default>\n' % a.xDefault)
        f.write('    </variable>\n')
    f.write('  </variables>\n')

    attNames = [x.name for x in model.domain.attributes]
    if needTransformation:
        f.write('  <transformations>\n')
        for a in model.domain.attributes:
            if getattr(a, 'xInputAsNumber', 0):
                f.write('    <variable>\n')
                f.write('    <id>%s</id>\n' % ('C'+XID(a.name)))
                f.write('    <name>%s</name>\n' % ('C'+XID(a.name)))
                f.write('    <hide>no</hide>\n')
                f.write('    <categorize>\n')
                f.write('      <from>%s</from>\n' % (XID(a.name)))
                f.write('      <cutoffs>%s</cutoffs>\n' % reduce(lambda x,y: x+';'+y, ["%3.1f" % i for i in a.getValueFrom.transformer.points]))
                f.write('    </categorize>\n')
                f.write('    </variable>\n')
                attNames[attNames.index(a.name)] = 'C'+XID(a.name)
        f.write('  </transformations>\n')

    f.write('  <models>\n')
    f.write('    <model>\n')
    f.write('      <name>%s</name>\n' % getattr(model, 'xModelName', 'Naive Bayesian Model'))
    f.write('      <variables>%s</variables>\n' % reduce(lambda x,y: x+';'+y, [XID(i) for i in attNames]))
    f.write('      <outcome>%s</outcome>\n' % XC(getattr(model, 'xOutcome', model.domain.classVar.name)))
    f.write('      <naivebayes>\n')
    f.write('        <class>%s</class>\n' % reduce(lambda x,y: x+','+y, [str(i) for i in model.conditionalDistributions.classes]))
    f.write('        <contingency>\n')
    
    for i in range(len(model.domain.attributes)):
        f.write('          <conditional attribute="%s">' % XID(attNames[i]))
        f.write("%s" % reduce(lambda x,y: x+";"+y, ["%s,%s" % (str(cd[0]),str(cd[1])) for cd in model.conditionalDistributions[i]]))
        f.write('</conditional>\n')
    f.write('        </contingency>\n')
    
    f.write('      </naivebayes>\n')
    f.write('    </model>\n')
    f.write('  </models>\n')
    f.write('</dsscheme>\n')
    

    if fopened:
        f.close()
