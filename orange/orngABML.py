# This module is used to handle argumented examples.

import Orange.core
import re
import string
import warnings

import Orange.classification.rules
import numpy
import math

# regular expressions
# exppression for testing validity of a set of arguments:
testVal = re.compile(r"""[" \s]*                              # remove any special characters at the beginning
                     ~?                                       # argument could be negative (~) or positive (without ~)  
                     {                                        # left parenthesis of the argument
                     \s*[\w\W]+                                # first attribute of the argument  
                     (\s*,\s*[\w\W]+)*                         # following attributes in the argument  
                     }                                        # right parenthesis of the argument  
                     (\s*,\s*~?{\s*[\w\W]+(\s*,\s*[\w\W]+)*})*  # following arguments
                     [" \s]*"""                               # remove any special characters at the end
                     , re.VERBOSE)

# splitting regular expressions
argRE = re.compile(r'[,\s]*(\{[^{}]+\})[,\s]*')
argAt = re.compile(r'[{},]+')
argCompare = re.compile(r'[(<=)(>=)<>]')

def strSign(oper):
    if oper == Orange.core.ValueFilter_continuous.Less:
        return "<"
    elif oper == Orange.core.ValueFilter_continuous.LessEqual:
        return "<="
    elif oper == Orange.core.ValueFilter_continuous.Greater:
        return ">"
    elif oper == Orange.core.ValueFilter_continuous.GreaterEqual:
        return ">="
    else: return "="

def strArg(arg, domain, leave_ref):
    if type(arg) == Orange.core.ValueFilter_discrete:
        return str(domain[arg.position].name)
    else:
        if leave_ref:
            return str(domain[arg.position].name)+strSign(arg.oper)
        else:
            return str(domain[arg.position].name)+strSign(arg.oper)+str(arg.ref)
            
def listOfAttributeNames(rule, leave_ref = False):
    if not rule.filter.conditions:
        return ""
    list = ""
    for val in rule.filter.conditions[:-1]:
        lr = leave_ref or val.unspecialized_condition
        list += strArg(val, rule.filter.domain, lr) + ","
    lr = leave_ref or rule.filter.conditions[-1].unspecialized_condition
    list += strArg(rule.filter.conditions[-1], rule.filter.domain, lr)
    return list
        
    
class Argumentation:
    """ Class that describes a set of positive and negative arguments
    this class is used as a value for ArgumentationVariable. """
    def __init__(self):
        self.positive_arguments = Orange.core.RuleList()
        self.negative_arguments = Orange.core.RuleList()
        self.not_yet_computed_arguments = [] # Arguments that need the whole data set
                                          # when processing are stored here 
    
    # add an argument that supports the class of the example
    def addPositive(self, argument):
        self.positive_arguments.append(argument)

    # add an argument that opposes the class of the example
    def addNegative(self, argument):
        self.negative_arguments.append(argument)

    def addNotYetComputed(self, argument, notyet, positive):
        self.not_yet_computed_arguments.append((argument,notyet,positive))

    def __str__(self):
        retValue = ""
        # iterate through positive arguments (rules) and
        # write them down as a text list
        if len(self.positive_arguments)>0:
            for (i,pos) in enumerate(self.positive_arguments[:-1]):
                retValue+="{"+listOfAttributeNames(pos)+"}"
                retValue+=","
            retValue+="{"+listOfAttributeNames(self.positive_arguments[-1])+"}"
            # do the same thing for negative argument,
            # just that this time use sign "~" in front of the list
        if len(self.negative_arguments)>0:
            if len(retValue)>0:
                retValue += ","
            for (i,neg) in enumerate(self.negative_arguments[:-1]):
                retValue+="~"
                retValue+="{"+listOfAttributeNames(neg,leave_ref=True)+"}"
                retValue+=","
            retValue+="~{"+listOfAttributeNames(self.negative_arguments[-1],leave_ref=True)+"}"
        return retValue

POSITIVE = True
NEGATIVE = False
class ArgumentVariable(Orange.core.PythonVariable):
    """ For writing and parsing arguments in .tab files. """
    def str2val(self, strV):
        """ convert str to val - used for creating variables. """
        return self.filestr2val(strV, None)
    
    def filestr2val(self, strV, example=None):
        """ write arguments (from string in file) to value - used also as a function for reading from data. """
        mt = testVal.match(strV)
        if not mt or not mt.end() == len(strV):
            warnings.warn(strV+" is a badly formed argument.")
            return Orange.core.PythonValueSpecial(2) # return special if argument doesnt match the formal form 

        if not example:
            example = self.example
        domain = example.domain

        # get a set of arguments
        splitedSet = filter(lambda x:x!='' and x!='"', argRE.split(strV))

        # create an Argumentation object - an empty set of arguments
        argumentation = Argumentation()
        type = POSITIVE # type of argument - positive = True / negative = False
        for sp in splitedSet:
            # for each argument determine whether it is positive or negative
            if sp == '~':
                type = NEGATIVE
                continue
            argument = Orange.core.Rule(filter=Orange.core.Filter_values(domain=domain))
            argument.setattr("unspecialized_argument",False)
            
            reasonValues = filter(lambda x:x!='' and x!='"', argAt.split(sp)) # reasons in this argument
            # iterate through argument names
            for r in reasonValues:
                r=string.strip(r)   # Remove all white characters on both sides
                try:
                    attribute = domain[r]
                except:
                    attribute = None
                if attribute: # only attribute name is mentioned as a reason
                    if domain.index(attribute)<0:
                        warnings.warn("Meta attribute %s used in argument. Is this intentional?"%r)
                        continue
                    value = example[attribute]
                    if attribute.varType == Orange.core.VarTypes.Discrete: # discrete argument
                        argument.filter.conditions.append(Orange.core.ValueFilter_discrete(
                                                            position = domain.attributes.index(attribute),
                                                            values=[value],
                                                            acceptSpecial = 0))
                        argument.filter.conditions[-1].setattr("unspecialized_condition",False)
                    else: # continuous but without reference point
                        warnings.warn("Continous attributes (%s) in arguments should not be used without a comparison sign (<,<=,>,>=)"%r)            
            
                else: # attribute and something name is the reason, probably cont. attribute
                    # one of four possible delimiters should be found, <,>,<=,>=
                    splitReason = filter(lambda x:x!='' and x!='"', argCompare.split(r))
                    if len(splitReason)>2 or len(splitReason)==0:
                        warnings.warn("Reason %s is a badly formed part of an argument."%r)
                        continue
                    # get attribute name and continous reference value
                    attributeName = string.strip(splitReason[0])
                    if len(splitReason) > 1:
                        refValue = string.strip(splitReason[1])
                    else:
                        refValue = ""
                    if refValue:
                        sign = r[len(attributeName):-len(refValue)]
                    else:
                        sign = r[len(attributeName):]

                    # evaluate name and value                        
                    try:
                        attribute = domain[attributeName]
                    except:
                        warnings.warn("Attribute %s is not a part of the domain"%attributeName)
                        continue
                    if domain.index(attribute)<0:
                        warnings.warn("Meta attribute %s used in argument. Is this intentional?"%r)
                        continue
                    if refValue:
                        try:
                            ref = eval(refValue)
                        except:
                            warnings.warn("Error occured while reading value by argument's reason. Argument: %s, value: %s"%(r,refValue))
                            continue
                    else:
                        ref = 0.                        
                    if sign == "<": oper = Orange.core.ValueFilter_continuous.Less
                    elif sign == ">": oper = Orange.core.ValueFilter_continuous.Greater
                    elif sign == "<=": oper = Orange.core.ValueFilter_continuous.LessEqual
                    else: oper = Orange.core.ValueFilter_continuous.GreaterEqual
                    argument.filter.conditions.append(Orange.core.ValueFilter_continuous(
                                                position = domain.attributes.index(attribute),
                                                oper=oper,
                                                ref=ref,
                                                acceptSpecial = 0))
                    if not refValue and type == POSITIVE:
                        argument.filter.conditions[-1].setattr("unspecialized_condition",True)
                        argument.setattr("unspecialized_argument",True)
                    else:
                        argument.filter.conditions[-1].setattr("unspecialized_condition",False)

            if example.domain.classVar:
                argument.classifier = Orange.core.DefaultClassifier(defaultVal = example.getclass())
            argument.complexity = len(argument.filter.conditions)

            if type: # and len(argument.filter.conditions):
                argumentation.addPositive(argument)
            else: # len(argument.filter.conditions):
                argumentation.addNegative(argument)
            type = POSITIVE
        return argumentation                    


    # used for writing to data: specify output (string) presentation of arguments in tab. file
    def val2filestr(self, val, example):
        return str(val)

    # used for writing to string
    def val2str(self, val):
        return str(val)        

    
class ArgumentFilter_hasSpecial:
    def __call__(self, examples, attribute, target_class=-1, negate=0):
        indices = [0]*len(examples)
        for i in range(len(examples)):
            if examples[i][attribute].isSpecial():
                indices[i]=1
            elif target_class>-1 and not int(examples[i].getclass()) == target_class:
                indices[i]=1
            elif len(examples[i][attribute].value.positive_arguments) == 0:
                indices[i]=1
        return examples.select(indices,0,negate=negate)

def evaluateAndSortArguments(examples, argAtt, evaluateFunction = None, apriori = None):
    """ Evaluate positive arguments and sort them by quality. """
    if not apriori:
        apriori = Orange.core.Distribution(examples.domain.classVar,examples)
    if not evaluateFunction:
        evaluateFunction = Orange.core.RuleEvaluator_Laplace()
        
    for e in examples:
        if not e[argAtt].isSpecial():
            for r in e[argAtt].value.positive_arguments:
                r.filterAndStore(examples, 0, e[examples.domain.classVar])
                r.quality = evaluateFunction(r,examples,0,int(e[examples.domain.classVar]),apriori)
            e[argAtt].value.positive_arguments.sort(lambda x,y: -cmp(x.quality, y.quality))

def isGreater(oper):
    if oper == Orange.core.ValueFilter_continuous.Greater or \
       oper == Orange.core.ValueFilter_continuous.GreaterEqual:
        return True
    return False

def isLess(oper):
    if oper == Orange.core.ValueFilter_continuous.Less or \
       oper == Orange.core.ValueFilter_continuous.LessEqual:
        return True
    return False

class ConvertClass:
    """ Converting class variables into dichotomous class variable. """
    def __init__(self, classAtt, classValue, newClassAtt):
        self.classAtt = classAtt
        self.classValue = classValue
        self.newClassAtt = newClassAtt

    def __call__(self,example, returnWhat):
        if example[self.classAtt] == self.classValue:
            return Orange.core.Value(self.newClassAtt, self.classValue+"_")
        else:
            return Orange.core.Value(self.newClassAtt, "not " + self.classValue)
    
def createDichotomousClass(domain, att, value, negate, removeAtt = None):
    # create new variable
    newClass = Orange.core.EnumVariable(att.name+"_", values = [str(value)+"_", "not " + str(value)])
    positive = Orange.core.Value(newClass, str(value)+"_")
    negative = Orange.core.Value(newClass, "not " + str(value))
    newClass.getValueFrom = ConvertClass(att,str(value),newClass)
    
    att = [a for a in domain.attributes]
    newDomain = Orange.core.Domain(att+[newClass])
    newDomain.addmetas(domain.getmetas())
    if negate==1:
        return (newDomain, negative)
    else:
        return (newDomain, positive)


class ConvertCont:
    def __init__(self, position, value, oper, newAtt):
        self.value = value
        self.oper = oper
        self.position = position
        self.newAtt = newAtt

    def __call__(self,example, returnWhat):
        if example[self.position].isSpecial():
            return example[self.position]
        if isLess(self.oper):
            if example[self.position]<self.value:
                return Orange.core.Value(self.newAtt, self.value)
            else:
                return Orange.core.Value(self.newAtt, float(example[self.position]))
        else:
            if example[self.position]>self.value:
                return Orange.core.Value(self.newAtt, self.value)
            else:
                return Orange.core.Value(self.newAtt, float(example[self.position]))


def addErrors(test_data, classifier):
    """ Main task of this function is to add probabilistic errors to examples. Function
        also computes classification accuracy as a by product."""
    correct        = 0.0
    clDistribution = orange.Distribution(test_data.domain.classVar)
    for ex_i, ex in enumerate(test_data):
        (cl,prob) = classifier(ex,orange.GetBoth)
        # add prob difference to ProbError
        if prob[ex.getclass()] < 0.01:
            print prob
        ex.setmeta("ProbError", ex.getmeta("ProbError") + 1.-prob[ex.getclass()]) 
        ex.setmeta("ProbErrorSD", ex.getmeta("ProbErrorSD") + math.pow(1.-prob[ex.getclass()],2))
        if test_data.domain.hasmeta("CoveredRules"):
            for r_i,r in enumerate(classifier.rules):
                if r(ex) and (not hasattr(cl, "ruleBetas") or cl.ruleBetas[r_i]):
                    ex.setmeta("CoveredRules", ex.getmeta("CoveredRules")+orngCN2.ruleToString(r)+";")
        if cl == ex.getclass(): # compute classification accuracy (just for fun)
            correct += 1.
            clDistribution[ex.getclass()] += 1.
    apriori = orange.Distribution(test_data.domain.classVar,test_data)
    correct /= len(test_data)
    for cl_v in test_data.domain.classVar:
        clDistribution[cl_v] /= max(apriori[cl_v],1)
    # add space between different test set in "covered rules"
    for ex in test_data:
        ex.setmeta("CoveredRules", ex.getmeta("CoveredRules")+";      ;")
    return (correct,clDistribution)


def nCrossValidation(data,learner,weightID=0,folds=5,n=4,gen=0):
    """ Function performs n x fold crossvalidation. For each classifier
        test set is updated by calling function addErrors. """
    acc = 0.0
    dist = orange.Distribution(data.domain.classVar)    
    pick = orange.MakeRandomIndicesCV(folds=folds, randseed=gen, stratified = orange.MakeRandomIndices.StratifiedIfPossible)    
    for d in data:
        d.setmeta("ProbError",0.)
        d.setmeta("ProbErrorSD",0.)
##        d.setmeta("Rules","")
    for n_i in range(n):
        pick.randseed = gen+10*n_i
        selection = pick(data)
        for folds_i in range(folds):
            for data_i,e in enumerate(data):
                try:
                    if e["Arguments"]: # examples with arguments do not need to be tested
                        selection[data_i]=folds_i+1
                except:
                    pass
            train_data = data.selectref(selection, folds_i,negate=1)
            test_data = data.selectref(selection, folds_i,negate=0)
            classifier = learner(train_data,weightID)
            classifier.setattr("data", train_data)
            acc1,dist1 = addErrors(test_data, classifier)

            print "N=%d, Folds=%d: %s %s" % (n_i+1, folds_i, acc1, dist1)
            acc += acc1
            for cl_i in range(len(data.domain.classVar.values)):
                dist[cl_i] += dist1[cl_i]

    for e in data:
        avgProb = e.getmeta("ProbError")/n
        sumSq = e.getmeta("ProbErrorSD")
        sumProb = e.getmeta("ProbError")
        if n>1 and (sumSq-2*avgProb*sumProb+n*math.pow(avgProb,2))/(n-1) > 0.:
            e.setmeta("ProbErrorSD", math.sqrt((sumSq-2*avgProb*sumProb+n*math.pow(avgProb,2))/(n-1)))
        else:
            e.setmeta("ProbErrorSD", 0.)
        e.setmeta("ProbError", avgProb)
    acc = acc/n/folds
    for cl_v in test_data.domain.classVar:
        dist[cl_v] /= n*folds
    return (acc,dist)


def findProb(learner,examples,weightID=0,folds=5,n=4,gen=0):
    """ General method for calling to find problematic example.
        It returns all examples along with average probabilistic errors.
        Taking the one with highest error is the same as taking the most
        problematic example. """
    newDomain = orange.Domain(examples.domain.attributes, examples.domain.classVar)
    newDomain.addmetas(examples.domain.getmetas())
    newExamples = orange.ExampleTable(newDomain, examples)
    if not newExamples.domain.hasmeta("ProbError"):
        newId = orange.newmetaid()
        newDomain.addmeta(newId, orange.FloatVariable("ProbError"))
        newExamples = orange.ExampleTable(newDomain, examples)
    if not newExamples.domain.hasmeta("ProbErrorSD"):
        newId = orange.newmetaid()
        newDomain.addmeta(newId, orange.FloatVariable("ProbErrorSD"))
        newExamples = orange.ExampleTable(newDomain, examples)
    if not newExamples.domain.hasmeta("CoveredRules"):
        newId = orange.newmetaid()
        newDomain.addmeta(newId, orange.StringVariable("CoveredRules"))
        newExamples = orange.ExampleTable(newDomain, examples)        
    if not newExamples.domain.hasmeta("SerialNumberPE"):
        newId = orange.newmetaid()
        newDomain.addmeta(newId, orange.FloatVariable("SerialNumberPE"))
        newExamples = orange.ExampleTable(newDomain, examples)
##        newExamples.domain.addmeta(newId, orange.FloatVariable("SerialNumberPE"))
##        newExamples.addMetaAttribute("SerialNumberPE", 0.)
        for i in range(len(newExamples)):
            newExamples[i]["SerialNumberPE"] = float(i)
    trs = nCrossValidation(newExamples,learner,weightID=weightID, folds=folds, n=n, gen=gen)
    return newExamples            
