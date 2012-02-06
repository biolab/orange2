import Orange

data = Orange.data.Table("lenses")

rc = Orange.classification.RandomClassifier()
rc.classVar = data.domain.classVar
rc.probabilities = [0.5, 0.3, 0.2]

for i in range(3):
    for ex in data[:5]:
        print ex, rc(ex)
    print