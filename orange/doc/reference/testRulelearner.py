import orange
import orngCN2

data = orange.ExampleTable("titanic.tab")

# create learner
learner = orange.RuleLearner()

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "*****"

learner.ruleFinder = orange.RuleBeamFinder()
learner.ruleFinder.evaluator = orngCN2.mEstimate(m=50)

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "****"

learner.ruleFinder.ruleStoppingValidator = orange.RuleValidator_LRS(alpha=0.01,min_coverage=10,max_rule_complexity = 2)
learner.ruleFinder.ruleFilter = orange.RuleBeamFilter_Width(width = 50)

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
