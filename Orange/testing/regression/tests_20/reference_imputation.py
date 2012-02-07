# Description: Shows the use of classes for imputation
# Category:    default classification accuracy, statistics
# Classes:     Imputer, Imputer_defaults, Imputer_asValue, Imputer_model, ImputerConstructor, ImputerConstructor_minimal, ImputerConstructor_maximal, ImputerConstructor_average, ImputerConstructor_asValue, ImputerConstructor_model
# Uses:        bridges
# Referenced:  imputation.htm

import orange

data = orange.ExampleTable("bridges")

print "\n*** IMPUTING MINIMAL VALUES ***\n"

imputer = orange.ImputerConstructor_minimal(data)

print "Example w/ missing values"
print data[19]
print "Imputed:"
print imputer(data[19])
print

impdata = imputer(data)
for i in range(20, 25):
    print data[i]
    print impdata[i]
    print


print "\n*** IMPUTING MAXIMAL VALUES ***\n"

imputer = orange.ImputerConstructor_maximal(data)

print "Example w/ missing values"
print data[19]
print "Imputed:"
print imputer(data[19])
print

impdata = imputer(data)
for i in range(20, 25):
    print data[i]
    print impdata[i]
    print


print "\n*** IMPUTING AVERAGE/MAJORITY VALUES ***\n"

imputer = orange.ImputerConstructor_average(data)

print "Example w/ missing values"
print data[19]
print "Imputed:"
print imputer(data[19])
print

impdata = imputer(data)
for i in range(20, 25):
    print data[i]
    print impdata[i]
    print


print "\n*** MANUALLY CONSTRUCTED IMPUTER ***\n"

imputer = orange.Imputer_defaults(data.domain)
imputer.defaults["LENGTH"] = 1234

print "Example w/ missing values"
print data[19]
print "Imputed:"
print imputer(data[19])
print

impdata = imputer(data)
for i in range(20, 25):
    print data[i]
    print impdata[i]
    print


print "\n*** TREE-BASED IMPUTATION ***\n"

import orngTree
imputer = orange.ImputerConstructor_model()
imputer.learnerContinuous = imputer.learnerDiscrete = orngTree.TreeLearner(minSubset = 20)
imputer = imputer(data)

print "Example w/ missing values"
print data[19]
print "Imputed:"
print imputer(data[19])
print


impdata = imputer(data)
for i in range(20, 25):
    print data[i]
    print impdata[i]
    print



print "\n*** BAYES and AVERAGE IMPUTATION ***\n"

imputer = orange.ImputerConstructor_model()
imputer.learnerContinuous = orange.MajorityLearner()
imputer.learnerDiscrete = orange.BayesLearner()
imputer = imputer(data)

print "Example w/ missing values"
print data[19]
print "Imputed:"
print imputer(data[19])
print


impdata = imputer(data)
for i in range(20, 25):
    print data[i]
    print impdata[i]
    print


print "\n*** CUSTOM IMPUTATION BY MODELS ***\n"

imputer = orange.Imputer_model()
imputer.models = [None] * len(data.domain)

imputer.models[data.domain.index("LANES")] = orange.DefaultClassifier(2.0)

tord = orange.DefaultClassifier(orange.Value(data.domain["T-OR-D"], "THROUGH"))
imputer.models[data.domain.index("T-OR-D")] = tord

import orngTree
len_domain = orange.Domain(["MATERIAL", "SPAN", "ERECTED", "LENGTH"], data.domain)
len_data = orange.ExampleTable(len_domain, data)
len_tree = orngTree.TreeLearner(len_data, minSubset=20)
imputer.models[data.domain.index("LENGTH")] = len_tree
orngTree.printTxt(len_tree)

spanVar = data.domain["SPAN"]
def computeSpan(ex, rw):
    if ex["TYPE"] == "WOOD" or ex["PURPOSE"] == "WALK":
        return orange.Value(spanVar, "SHORT")
    else:
        return orange.Value(spanVar, "MEDIUM")

imputer.models[data.domain.index("SPAN")] = computeSpan

for i in range(20, 25):
    print data[i]
    print impdata[i]
    print

##for i in imputer(data):
##    print i


print "\n*** IMPUTATION WITH SPECIAL VALUES ***\n"

imputer = orange.ImputerConstructor_asValue(data)

original = data[19]
imputed = imputer(data[19])

print original.domain
print
print imputed.domain
print

for i in original.domain:
    print "%s: %s -> %s" % (original.domain[i].name, original[i], imputed[i.name]),
    if original.domain[i].varType == orange.VarTypes.Continuous:
        print "(%s)" % imputed[i.name+"_def"]
    else:
        print
print

impdata = imputer(data)
for i in range(20, 25):
    print data[i]
    print impdata[i]
    print
