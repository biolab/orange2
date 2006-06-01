# Provided only for backward compatibility.
# Don't use

import orange

def build(data, support=0.9, maxItemSets=15000):
    return AssociationRulesInducer(data, support=support, maxItemSets=maxItemSets)

AssociationRulesInduces = orange.AssociationRulesInducer
