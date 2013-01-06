import Orange

data = Orange.data.Table("promoters")
gain = Orange.feature.scoring.InfoGain()
best = [f for _, f in sorted((gain(x, data), x) for x in data.domain.features)[-5:]]
print "Features:", len(data.domain.features)
print "Best ones:", ", ".join([x.name for x in best])