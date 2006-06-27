# Provided only for backward compatibility.
# Don't use

import orange

def build(data, support=0.9, maxItemSets=15000):
    return AssociationRulesInducer(data, support=support, maxItemSets=maxItemSets)

AssociationRulesInduces = orange.AssociationRulesInducer

def printRules(rules, ms = []):
    if ms:
        print "\t".join([m[:4] for m in ms]) + "\trule"
        for rule in rules:
            print "\t".join(["%5.3f" % getattr(rule, m) for m in ms]) + "\t" + str(rule)
    else:
        for rule in rules:
            print rule

class Cmp:
    def __init__(self, ms):
        self.ms = ms

    def __call__(self, r1, r2):        
        for m in self.ms:
            c = -cmp(getattr(r1, m), getattr(r2, m))
            if c:
                return c
        return 0

def sort(rules, ms = ["support"]):
    rules.sort(Cmp(ms))