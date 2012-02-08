# Description: Shows the use of classes for imputation
# Category:    imputation
# Uses:        bridges
# Referenced:  Orange.feature.html#imputation
# Classes:     Orange.feature.imputation.Imputer, Orange.feature.imputation.Imputer_defaults, Orange.feature.imputation.Imputer_asValue, Orange.feature.imputation.Imputer_model, Orange.feature.imputation.ImputerConstructor, Orange.feature.imputation.ImputerConstructor_minimal, Orange.feature.imputation.ImputerConstructor_maximal, Orange.feature.imputation.ImputerConstructor_average, Orange.feature.imputation.ImputerConstructor_asValue, Orange.feature.imputation.ImputerConstructor_model

import Orange

bridges = Orange.data.Table("bridges")

print "*** IMPUTING MINIMAL VALUES ***"
imputer = Orange.feature.imputation.MinimalConstructor(bridges)
print "Example w/ missing values"
print bridges[19]
print "Imputed:"
print imputer(bridges[19])
print

impdata = imputer(bridges)
for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print


print "*** IMPUTING MAXIMAL VALUES ***"
imputer = Orange.feature.imputation.MaximalConstructor(bridges)
print "Example w/ missing values"
print bridges[19]
print "Imputed:"
print imputer(bridges[19])
print

impdata = imputer(bridges)
for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print


print "*** IMPUTING AVERAGE/MAJORITY VALUES ***"
imputer = Orange.feature.imputation.AverageConstructor(bridges)
print "Example w/ missing values"
print bridges[19]
print "Imputed:"
print imputer(bridges[19])
print

impdata = imputer(bridges)
for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print


print "*** MANUALLY CONSTRUCTED IMPUTER ***"
imputer = Orange.feature.imputation.Defaults(bridges.domain)
imputer.defaults["LENGTH"] = 1234
print "Example w/ missing values"
print bridges[19]
print "Imputed:"
print imputer(bridges[19])
print

impdata = imputer(bridges)
for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print


print "*** TREE-BASED IMPUTATION ***"

imputer = Orange.feature.imputation.ModelConstructor()
imputer.learner_continuous = imputer.learner_discrete = Orange.classification.tree.TreeLearner(minSubset=20)
imputer = imputer(bridges)
print "Example w/ missing values"
print bridges[19]
print "Imputed:"
print imputer(bridges[19])
print

impdata = imputer(bridges)
for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print


print "*** BAYES and AVERAGE IMPUTATION ***"
imputer = Orange.feature.imputation.ModelConstructor()
imputer.learner_continuous = Orange.regression.mean.MeanLearner()
imputer.learner_discrete = Orange.classification.bayes.NaiveLearner()
imputer = imputer(bridges)
print "Example w/ missing values"
print bridges[19]
print "Imputed:"
print imputer(bridges[19])
print
impdata = imputer(bridges)
for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print


print "*** CUSTOM IMPUTATION BY MODELS ***"
imputer = Orange.feature.imputation.Model()
imputer.models = [None] * len(bridges.domain)
imputer.models[bridges.domain.index("LANES")] = Orange.classification.ConstantClassifier(2.0)
tord = Orange.classification.ConstantClassifier(Orange.data.Value(bridges.domain["T-OR-D"], "THROUGH"))
imputer.models[bridges.domain.index("T-OR-D")] = tord


len_domain = Orange.data.Domain(["MATERIAL", "SPAN", "ERECTED", "LENGTH"], bridges.domain)
len_data = Orange.data.Table(len_domain, bridges)
len_tree = Orange.classification.tree.TreeLearner(len_data, minSubset=20)
imputer.models[bridges.domain.index("LENGTH")] = len_tree
print len_tree

span_var = bridges.domain["SPAN"]
def compute_span(ex, rw):
    if ex["TYPE"] == "WOOD" or ex["PURPOSE"] == "WALK":
        return Orange.data.Value(span_var, "SHORT")
    else:
        return Orange.data.Value(span_var, "MEDIUM")

imputer.models[bridges.domain.index("SPAN")] = compute_span

for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print


print "*** IMPUTATION WITH SPECIAL VALUES ***"
imputer = Orange.feature.imputation.AsValueConstructor(bridges)
original = bridges[19]
imputed = imputer(bridges[19])
print original.domain
print
print imputed.domain
print

for i in original.domain:
    print "%s: %s -> %s" % (original.domain[i].name, original[i], imputed[i.name]),
    if original.domain[i].var_type == Orange.feature.Type.Continuous:
        print "(%s)" % imputed[i.name+"_def"]
    else:
        print
print

impdata = imputer(bridges)
for i in range(20, 25):
    print bridges[i]
    print impdata[i]
    print
