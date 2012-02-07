# Description: Demonstrates the use of orngCN2 rules
# Category:    classification, rules
# Classes:     CN2Learner, CN2UnorderedLearner, CN2UnorderedLearner
# Uses:        titanic.tab

import orange
import orngCN2

data = orange.ExampleTable("titanic.tab")

# create learner
learner = orngCN2.CN2Learner()

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "*****"

learner = orngCN2.CN2UnorderedLearner()

learner.ruleFinder = orange.RuleBeamFinder()
learner.ruleFinder.evaluator = orngCN2.mEstimate(m=50)

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "****"

learner = orngCN2.CN2SDUnorderedLearner()

learner.ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha=0.01,min_coverage=10,max_rule_complexity = 2)
learner.ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width = 50)

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "****"

learner = orngCN2.CN2UnorderedLearner()

learner.ruleFinder = orange.RuleBeamFinder()
learner.ruleFinder.evaluator = orngCN2.WRACCEvaluator()

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "****"


