# Provided only for backward compatibility.

import orange

def build(data, support=0.9, maxItemSets=15000):
    return AssociationRulesInducer(data, support=support, maxItemSets=maxItemSets)

AssociationRulesInduces = orange.AssociationRulesInducer

import Orange.associate

printRules = Orange.associate.print_rules
sort = Orange.associate.sort
