# Description: Customized Rule Induction on Titanic dataset
# Category:    classification
# Uses:        titanic
# Referenced:  Orange.classification.rules
# Classes:     Orange.classification.rules.RuleLearner, Orange.classification.rules.RuleBeamFinder, Orange.classification.rules.RuleValidator_LRS, Orange.classification.rules.RuleBeamFilter_Width

import Orange

learner = Orange.classification.rules.RuleLearner()
learner.rule_finder = Orange.classification.rules.RuleBeamFinder()
learner.rule_finder.evaluator = Orange.classification.rules.MEstimateEvaluator(m=50)

titanic =  Orange.data.Table("titanic")
classifier = learner(titanic)

for r in classifier.rules:
    print Orange.classification.rules.rule_to_string(r)

learner.rule_finder.rule_stopping_validator = \
    Orange.classification.rules.RuleValidator_LRS(alpha=0.01,
                             min_coverage=10, max_rule_complexity = 2)
learner.rule_finder.rule_filter = \
    Orange.classification.rules.RuleBeamFilter_Width(width = 50)

classifier = learner(titanic)

for r in classifier.rules:
    print Orange.classification.rules.rule_to_string(r)
