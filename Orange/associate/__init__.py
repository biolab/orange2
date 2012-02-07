from orange import \
    AssociationRule, \
    AssociationRules, \
    AssociationRulesInducer, \
    AssociationRulesSparseInducer, \
    ItemsetNodeProxy, \
    ItemsetsSparseInducer

def print_rules(rules, ms = []):
    """
    Print the rules. If ms is left empty, only the rules are printed. If ms
    contains rules' attributes, e.g. ``["support", "confidence"]``, these are printed out as well.
    """
    if ms:
        print "\t".join([m[:4] for m in ms]) + "\trule"
        for rule in rules:
            print "\t".join(["%5.3f" % getattr(rule, m) for m in ms]) + "\t" + str(rule)
    else:
        for rule in rules:
            print rule

class __Cmp:
    def __init__(self, ms):
        self.ms = ms

    def __call__(self, r1, r2):        
        for m in self.ms:
            c = -cmp(getattr(r1, m), getattr(r2, m))
            if c:
                return c
        return 0

def sort(rules, ms = ["support"]):
    """
    Sort the rules according to the given criteria. The default key is "support"; list multiple keys in a list.
    """
    rules.sort(__Cmp(ms))
