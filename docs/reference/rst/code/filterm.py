import Orange

data = Orange.data.Table("inquisition")

surprised = Orange.data.filter.HasMeta(data, id=data.domain.index("surprise"))
for i in surprised:
    print i