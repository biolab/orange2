import Orange.data
import Orange.feature

data = Orange.data.Table("monks-1")

e1 = Orange.feature.Continuous("e=1")
e1.getValueFrom = Orange.core.ClassifierFromVar(whichVar=data.domain["e"])
e1.getValueFrom.transformer = Orange.core.Discrete2Continuous()
