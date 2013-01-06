import Orange

data = Orange.data.Table("voting.tab")
for x in data.domain.features:
    n_miss = sum(1 for d in data if d[x].is_special())
    print "%4.1f%% %s" % (100.*n_miss/len(data), x.name)
