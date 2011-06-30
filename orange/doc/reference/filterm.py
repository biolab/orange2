import orange

data = orange.ExampleTable("inquisition")

haveSurprise = orange.Filter_hasMeta(data, id = data.domain.index("surprise"))
for ex in haveSurprise:
    print ex