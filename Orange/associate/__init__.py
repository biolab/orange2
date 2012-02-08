from orange import \
    AssociationRule, \
    AssociationRules, \
    AssociationRulesInducer, \
    AssociationRulesSparseInducer, \
    ItemsetNodeProxy, \
    ItemsetsSparseInducer

def print_rules(rules, ms = []):
    """
    Print the association rules.

    :param rules: Association rules.
    :type rules: AssociationRules

    :param ms: Attributes of the rule to be printed with the rule (default: []). To report on confidence and support
     use ``ms=["support", "confidence"]``
    :type ms: list
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
    Sort the rules according to the given criteria.

    :param rules: Association rules.
    :type rules: AssociationRules

    :param ms: Sort keys (list of association rules' attributes, default: `["support"]`.
    :type ms: list
    """
    rules.sort(__Cmp(ms))
