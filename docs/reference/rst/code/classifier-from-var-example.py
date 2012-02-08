import Orange

monks = Orange.data.Table("monks-1")
e = monks.domain["e"]
e1 = Orange.feature.Discrete("e1", values = ["1", "not 1"])

def eTransformer(value):
    if int(value) == 0:
        return 0
    else:
        return 1

e1.getValueFrom = Orange.classification.ClassifierFromVar()
e1.getValueFrom.whichVar = e
e1.getValueFrom.transformer = eTransformer

monks2 = monks.select(["a", "b", "e", e1, "y"])
for i in monks2:
    print i

e1.getValueFrom = Orange.classification.ClassifierFromVarFD()
e1.getValueFrom.domain = monks.domain
e1.getValueFrom.position = monks.domain.attributes.index(e)
e1.getValueFrom.transformer = eTransformer

