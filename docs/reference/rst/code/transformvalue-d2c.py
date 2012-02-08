import Orange.data

monks = Orange.data.Table("monks-1")

e1 = Orange.feature.Continuous("e=1")
e1.getValueFrom = Orange.classification.ClassifierFromVar(whichVar=monks.domain["e"])
e1.getValueFrom.transformer = Orange.data.utils.Discrete2Continuous()

