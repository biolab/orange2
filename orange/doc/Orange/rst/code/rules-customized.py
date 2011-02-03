# Description: Customized Rule Induction on Titanic dataset
# Category:    classification
# Uses:        titanic
# Referenced:  Orange.classification.rules
# Classes:     Orange.classification.rules.RuleLearner, Orange.classification.rules.RuleBeamFinder, Orange.classification.rules.RuleValidator_LRS, Orange.classification.rules.RuleBeamFilter_Width

import Orange

learner = Orange.classification.rules.RuleLearner()
learner.ruleFinder = Orange.classification.rules.RuleBeamFinder()
learner.ruleFinder.evaluator = Orange.classification.rules.MEstimateEvaluator(m=50)

table =  Orange.data.Table("titanic")
classifier = learner(table)

for r in classifier.rules:
    print Orange.classification.rules.ruleToString(r)

learner.ruleFinder.ruleStoppingValidator = \
    Orange.classification.rules.RuleValidator_LRS(alpha=0.01,
                             min_coverage=10, max_rule_complexity = 2)
learner.ruleFinder.ruleFilter = \
    Orange.classification.rules.RuleBeamFilter_Width(width = 50)

classifier = learner(table)

for r in classifier.rules:
    print Orange.classification.rules.ruleToString(r)
