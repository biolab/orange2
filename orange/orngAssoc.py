import orange, string, types

def build(data, support=0.9, maxItemSets=15000):
    rules=AssociationRulesInducer(data, supp=support, maxItemSets=maxItemSets)
    AssociationRules.sortByFields(rules, ['support', 'confidence'])
    return rules

class AssociationRules(orange.AssociationRules):
    def sortByFields(self, measures):
        self.sort(lambda x,y: reduce(lambda L, R: L or R, [(-cmp(getattr(x, measure), getattr(y, measure))) for measure in measures if getattr(self[0], measure)], 0))

    def filterBySupport(self, limit):
        return self.filter(lambda rule: rule.support>limit)

    def filterByConfidence(self, limit):
        return self.filter(lambda rule: rule.confidence>limit)

    def sortByConfidence(self):
        self.sort(lambda a,b: -cmp(a.confidence, b.confidence))

    def sortBySupport(self):
        self.sort(lambda a,b: -cmp(a.support, b.support))

    def clone(self): # native
        return AssociationRules(self)

    def __side2string(side):
        rs=''
        for i in side:
            if not i.isSpecial():
                if len(rs):
                    rs += ", "
                rs += "%s='%s'" % (i.variable.name, i.native())
        return rs

    def __rule2string(asrule):
        return "%s  =>  %s" % (__side2string(asrule.left), __side2string(asrule.right))

    def printRules(rules):
        out = ""
        for rule in rules:
            out += str(rule)+"\n"
        return out

    def printMeasures(self, ms):
        if ms!=[]:
            print string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%s' % (m[0:4]) for m in ms])),"\trule"
        for rule in self:
            L=string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%.3f' % getattr(rule,m) for m in ms],""))
            if L!="":
                print L,'\t',rule
            else:
                print rule

class AssociationRulesInducer(orange.AssociationRulesInducer):
    __call_construction_type = AssociationRules
    pass

def __f(rule, ms):
    if ms!=[]:
         print string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%s' % (m[0:4]) for m in ms]))+"\trule"
    L=string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%.3f' % getattr(rule,m) for m in ms],""))
    if L!="":
        print L+'\t',rule
    else:
        print rule
    
orange.__addmethod(orange.AssociationRule, "printMeasures", __f)

def __g(rule, ms):
    s=""
    if ms!=[]:
         s += string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%s' % (m[0:4]) for m in ms]))+"\trule\n"
    L=string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%.3f' % getattr(rule,m) for m in ms],""))
    if L!="":
        s += L+'\t'+`rule`
    else:
        s += rule
    s+='\n'
    return s

if not orange.AssociationRule.__dict__.has_key("__output_measures"):
    orange.setoutput(orange.AssociationRule, "measures", __g)