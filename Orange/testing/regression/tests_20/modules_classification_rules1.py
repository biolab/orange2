import Orange
import orngCN2

data = Orange.data.Table("titanic.tab")

# create learner
learner = Orange.core.RuleLearner()

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "*****"

learner.ruleFinder = Orange.core.RuleBeamFinder()
learner.ruleFinder.evaluator = orngCN2.mEstimate(m=50)

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "****"

learner.ruleFinder.ruleStoppingValidator = Orange.core.RuleValidator_LRS(alpha=0.01,min_coverage=10,max_rule_complexity = 2)
learner.ruleFinder.ruleFilter = Orange.core.RuleBeamFilter_Width(width = 50)

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
