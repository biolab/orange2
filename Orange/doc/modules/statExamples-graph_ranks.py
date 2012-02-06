import orange, orngStat

names = ["first", "third", "second", "fourth" ]
avranks =  [1.9, 3.2, 2.8, 3.3 ] 
cd = orngStat.compute_CD(avranks, 30) #tested on 30 datasets
orngStat.graph_ranks("statExamples-graph_ranks1.png", avranks, names, \
    cd=cd, width=6, textspace=1.5)
