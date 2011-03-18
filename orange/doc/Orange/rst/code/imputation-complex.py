# Description: Shows the use of classes for imputation
# Category:    imputation
# Uses:        bridges
# Referenced:  Orange.feature.html#imputation
# Classes:     Orange.feature.imputation.Imputer, Orange.feature.imputation.Imputer_defaults, Orange.feature.imputation.Imputer_asValue, Orange.feature.imputation.Imputer_model, Orange.feature.imputation.ImputerConstructor, Orange.feature.imputation.ImputerConstructor_minimal, Orange.feature.imputation.ImputerConstructor_maximal, Orange.feature.imputation.ImputerConstructor_average, Orange.feature.imputation.ImputerConstructor_asValue, Orange.feature.imputation.ImputerConstructor_model

import Orange

table = Orange.data.Table("bridges")

print "*** IMPUTING MINIMAL VALUES ***"
imputer = Orange.feature.imputation.ImputerConstructor_minimal(table)
print "Example w/ missing values"
print table[19]
print "Imputed:"
print imputer(table[19])
print

impdata = imputer(table)
for i in range(20, 25):
    print table[i]
    print impdata[i]
    print


print "*** IMPUTING MAXIMAL VALUES ***"
imputer = Orange.feature.imputation.ImputerConstructor_maximal(table)
print "Example w/ missing values"
print table[19]
print "Imputed:"
print imputer(table[19])
print

impdata = imputer(table)
for i in range(20, 25):
    print table[i]
    print impdata[i]
    print


print "*** IMPUTING AVERAGE/MAJORITY VALUES ***"
imputer = Orange.feature.imputation.ImputerConstructor_average(table)
print "Example w/ missing values"
print table[19]
print "Imputed:"
print imputer(table[19])
print

impdata = imputer(table)
for i in range(20, 25):
    print table[i]
    print impdata[i]
    print


print "*** MANUALLY CONSTRUCTED IMPUTER ***"
imputer = Orange.feature.imputation.Imputer_defaults(table.domain)
imputer.defaults["LENGTH"] = 1234
print "Example w/ missing values"
print table[19]
print "Imputed:"
print imputer(table[19])
print

impdata = imputer(table)
for i in range(20, 25):
    print table[i]
    print impdata[i]
    print


print "*** TREE-BASED IMPUTATION ***"

imputer = Orange.feature.imputation.ImputerConstructor_model()
imputer.learner_continuous = imputer.learner_discrete = Orange.classification.tree.TreeLearner(minSubset=20)
imputer = imputer(table)
print "Example w/ missing values"
print table[19]
print "Imputed:"
print imputer(table[19])
print

impdata = imputer(table)
for i in range(20, 25):
    print table[i]
    print impdata[i]
    print


print "*** BAYES and AVERAGE IMPUTATION ***"
imputer = Orange.feature.imputation.ImputerConstructor_model()
imputer.learner_continuous = Orange.regression.mean.MeanLearner()
imputer.learner_discrete = Orange.classification.bayes.NaiveLearner()
imputer = imputer(table)
print "Example w/ missing values"
print table[19]
print "Imputed:"
print imputer(table[19])
print
impdata = imputer(table)
for i in range(20, 25):
    print table[i]
    print impdata[i]
    print


print "*** CUSTOM IMPUTATION BY MODELS ***"
imputer = Orange.feature.imputation.Imputer_model()
imputer.models = [None] * len(table.domain)
imputer.models[table.domain.index("LANES")] = Orange.classification.ConstantClassifier(2.0)
tord = Orange.classification.ConstantClassifier(Orange.data.Value(table.domain["T-OR-D"], "THROUGH"))
imputer.models[table.domain.index("T-OR-D")] = tord


len_domain = Orange.data.Domain(["MATERIAL", "SPAN", "ERECTED", "LENGTH"], table.domain)
len_data = Orange.data.Table(len_domain, table)
len_tree = Orange.classification.tree.TreeLearner(len_data, minSubset=20)
imputer.models[table.domain.index("LENGTH")] = len_tree
print len_tree

span_var = table.domain["SPAN"]
def compute_span(ex, rw):
    if ex["TYPE"] == "WOOD" or ex["PURPOSE"] == "WALK":
        return orange.Value(span_var, "SHORT")
    else:
        return orange.Value(span_var, "MEDIUM")

imputer.models[table.domain.index("SPAN")] = compute_span

for i in range(20, 25):
    print table[i]
    print impdata[i]
    print


print "*** IMPUTATION WITH SPECIAL VALUES ***"
imputer = Orange.feature.imputation.ImputerConstructor_asValue(table)
original = table[19]
imputed = imputer(table[19])
print original.domain
print
print imputed.domain
print

for i in original.domain:
    print "%s: %s -> %s" % (original.domain[i].name, original[i], imputed[i.name]),
    if original.domain[i].varType == Orange.core.VarTypes.Continuous:
        print "(%s)" % imputed[i.name+"_def"]
    else:
        print
print

impdata = imputer(table)
for i in range(20, 25):
    print table[i]
    print impdata[i]
    print
